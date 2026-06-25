"""Regression tests for the Greene-residue + iota-profile certificate metrics.

This pins the *observable contract* of ``residue_iota_metrics`` and its
confinement-boundary reduction -- the PRIMARY field-line-topology certification
verdict -- not the internal steps:

  * Full pipeline (``test_residue_iota_metrics_*``): on a cheap analytic field
    with genuine radial shear (``RadialShearTransformField``, the same fixture the
    Greene-residue conventions suite uses), iota(r) is sampled across the band and
    rises monotonically through the realized 1/2 resonance; that in-band crossing
    is selected with a bracketing radial window; the residue probe converges on it
    on both O and X branches with an area-preserving tangent map (detM~1); and the
    ``residue_classification`` the module surfaces is exactly what
    ``classify_greene_residue`` assigns to the realized residue (the module does not
    re-label the probe's verdict). The empty-band path returns a well-formed
    zero-resonance aggregate.

  * Boundary reduction (``test_confinement_boundary_*``): ``confinement_boundary_
    from_residues`` classifies each island by its ELLIPTIC O-point residue and maps
    the canonical values an intact elliptic island (0<R<1) and a destroyed
    hyperbolic chain (R<0 or R>1) produce -- the values the conventions suite proves
    a real elliptic/hyperbolic orbit yields -- to the KAM confinement edge: the
    smallest radial label where an intact O-point gives way to a destroyed one, plus
    ``confined_core`` and the core/edge classes. The hyperbolic X-saddle of an intact
    island must NOT count as a torn chain (a real defect this pins); non-converged
    O-branches do not vote; an all-intact profile has a confined core and no boundary.

No test mirrors the implementation: each asserts a consequence a downstream cert
consumer reads (selected resonance, converged residue, classification agreement,
the confinement edge), and the boundary cases are proven against a synthetic
residue ladder with a known transition the reduction must find.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.topology.fieldline_map import (
    FieldlineIntegratorOptions,
)
from examples.single_stage_optimization.banana_opt.topology.greene_residue import (
    GREENE_RESIDUE_ELLIPTIC_O,
    GREENE_RESIDUE_HYPERBOLIC_X,
    classify_greene_residue,
)
from examples.single_stage_optimization.banana_opt.topology.iota_profile import (
    IotaProfile,
    IotaProfileSample,
)
from examples.single_stage_optimization.banana_opt.topology.periodic_orbit import (
    BRANCH_STATUS_CONVERGED,
    PeriodicOrbitSolverOptions,
)
from examples.single_stage_optimization.banana_opt.topology.rational_target import (
    GREENE_BRANCH_O,
    GREENE_BRANCH_X,
)
from examples.single_stage_optimization.banana_opt.topology.residue_iota_metrics import (
    RESIDUE_CLASS_BOUNDARY,
    RESIDUE_CLASS_DESTROYED,
    RESIDUE_CLASS_INTACT,
    _in_band_low_order_crossings,
    confinement_boundary_from_residues,
    residue_iota_metrics,
)


# --------------------------------------------------------------------------- #
# Cheap, exactly-known analytic vacuum fields. These mirror the fixtures in the
# Greene-residue conventions suite (test_greene_residue_conventions.py); they are
# inlined here rather than imported so this test stays self-contained and does not
# drag in that module's heavy top-level import block to borrow two field classes.
# Each is a pure rotation about (axis_r, axis_z): B_phi=1 and the (R,Z) velocity is
# the rigid/sheared poloidal rotation, so the realized iota is analytically known.
# --------------------------------------------------------------------------- #
def _finite_difference_dB_by_dX(evaluate_field, points):
    epsilon = 1.0e-6
    jacobian = np.empty((points.shape[0], 3, 3), dtype=float)
    for coordinate_index in range(3):
        direction = np.zeros(3, dtype=float)
        direction[coordinate_index] = epsilon
        plus = evaluate_field(points + direction)
        minus = evaluate_field(points - direction)
        jacobian[:, coordinate_index, :] = (plus - minus) / (2.0 * epsilon)
    return jacobian


class CircularTransformField:
    """Rigid poloidal rotation at constant rotational transform ``iota``."""

    def __init__(self, *, axis_r, axis_z, iota):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self.points = np.empty((0, 3), dtype=float)

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)
        return self

    def _B_at(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        cos_phi, sin_phi = x / radius, y / radius
        d_radius_dphi = -self.iota * (z - self.axis_z)
        d_z_dphi = self.iota * (radius - self.axis_r)
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)

    def B(self):
        return self._B_at(self.points)

    def dB_by_dX(self):
        return _finite_difference_dB_by_dX(self._B_at, self.points)


class RadialShearTransformField(CircularTransformField):
    """Poloidal rotation whose transform shears linearly with minor radius.

    iota(minor_radius) = reference_iota + shear*(minor_radius - reference_minor_radius).
    """

    def __init__(self, *, axis_r, axis_z, reference_iota, reference_minor_radius, shear):
        super().__init__(axis_r=axis_r, axis_z=axis_z, iota=reference_iota)
        self.reference_minor_radius = float(reference_minor_radius)
        self.shear = float(shear)

    def _B_at(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radius = np.sqrt(x**2 + y**2)
        minor_radius = np.sqrt((radius - self.axis_r) ** 2 + (z - self.axis_z) ** 2)
        cos_phi, sin_phi = x / radius, y / radius
        geometric_rate = self.iota + self.shear * (
            minor_radius - self.reference_minor_radius
        )
        d_radius_dphi = -geometric_rate * (z - self.axis_z)
        d_z_dphi = geometric_rate * (radius - self.axis_r)
        b_phi = np.ones_like(radius)
        b_r = d_radius_dphi / radius
        b_z = d_z_dphi / radius
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.stack([b_x, b_y, b_z], axis=-1)


# Analytic-field integrator/solver: the fields are exact rotations (no Biot-Savart
# cost), so tight tolerances converge in a few turns. Mirrors the conventions
# suite's settings so the residue numbers are directly comparable.
_ANALYTIC_INTEGRATOR = FieldlineIntegratorOptions(
    rtol=1.0e-10,
    atol=1.0e-12,
    max_step=0.05,
    samples_per_full_torus=64,
)
_ANALYTIC_SOLVER = PeriodicOrbitSolverOptions(
    residual_tolerance=1.0e-9,
    winding_tolerance=1.0e-6,
    max_iterations=8,
    max_step_norm=0.08,
)
# Four poloidal phases spanning the section so the O and X periodic orbits of the
# 1/2 chain are both in the multistart scan.
_FOUR_PHASES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)


def _shear_field_1_over_2():
    """Analytic field whose realized iota rises 0.45->0.55 through the 1/2 surface.

    ``RadialShearTransformField`` rotates at ``reference_iota + shear*(r - r_ref)``;
    with reference_iota=0.5 at minor radius 0.2 and shear 0.5, a midplane seed at
    label L winds at iota = 0.5 + 0.5*(L - 0.2), so iota = 1/2 is realized exactly
    at label 0.2 (inside the [0.1, 0.3] band traced below).
    """

    return RadialShearTransformField(
        axis_r=1.0,
        axis_z=0.0,
        reference_iota=0.5,
        reference_minor_radius=0.2,
        shear=0.5,
    )


def _metrics_on_shear_field():
    """``residue_iota_metrics`` on the 1/2 shear field over the [0.1, 0.3] band."""

    return residue_iota_metrics(
        _shear_field_1_over_2(),
        axis_r=1.0,
        axis_z=0.0,
        nfp=1,
        radial_lo=0.1,
        radial_hi=0.3,
        radial_label_count=11,
        max_denominator=6,
        toroidal_turns=20,
        iota_integrator_options=_ANALYTIC_INTEGRATOR,
        integrator_options=_ANALYTIC_INTEGRATOR,
        solver_options=_ANALYTIC_SOLVER,
        phase_angles=_FOUR_PHASES,
        max_targets=1,
    )


def test_residue_iota_metrics_samples_monotone_iota_profile_across_band():
    """iota(r) is traced at every seed and rises monotonically through the band."""
    metrics = _metrics_on_shear_field()
    profile = metrics["iota_profile"]

    assert profile["n_labels"] == 11
    assert profile["n_valid"] == 11  # an exact analytic field never fails a trace
    assert profile["n_failed"] == 0
    iotas = profile["iota"]
    assert all(value is not None for value in iotas)
    # Realized iota = 0.5 + 0.5*(label - 0.2): edges 0.45 (label 0.1) and 0.55
    # (label 0.3), strictly increasing in label.
    assert iotas[0] == pytest.approx(0.45, abs=1.0e-6)
    assert iotas[-1] == pytest.approx(0.55, abs=1.0e-6)
    assert profile["iota_min"] == pytest.approx(0.45, abs=1.0e-6)
    assert profile["iota_max"] == pytest.approx(0.55, abs=1.0e-6)
    assert profile["iota_edge"] == pytest.approx(0.55, abs=1.0e-6)
    assert all(b > a for a, b in zip(iotas, iotas[1:]))


def test_residue_iota_metrics_selects_in_band_half_resonance_with_bracket():
    """Only the realized in-band 1/2 crossing is probed, with a bracketing window."""
    metrics = _metrics_on_shear_field()
    crossings = metrics["rational_crossings"]

    assert len(crossings) == 1
    crossing = crossings[0]
    assert (crossing["p"], crossing["q"]) == (1, 2)
    assert crossing["iota"] == pytest.approx(0.5, abs=1.0e-6)
    # The resonance sits at label 0.2 and is bracketed by adjacent traced samples.
    assert crossing["radial_label"] == pytest.approx(0.2, abs=1.0e-3)
    assert crossing["bracket_lower_label"] < crossing["radial_label"]
    assert crossing["bracket_upper_label"] > crossing["radial_label"]
    # iota_band the module derived from the profile: bounded above by 1 (no
    # near-axis winding>1 branch exists on this monotone sub-1 field).
    band_lo, band_hi = metrics["iota_band_probed"]
    assert band_lo == pytest.approx(0.45, abs=1.0e-6)
    assert band_hi == pytest.approx(0.55, abs=1.0e-6)
    assert crossing["p"] == metrics["aggregate"]["residues"][0]["p"]


# A slid_clean-shaped profile: full-torus winding reads iota>1 in the near-axis
# integer-offset ring (label 0.01), then the PHYSICAL n=0 branch falls monotonically
# toward the iota->0 X-point separatrix. This is the profile whose raw (unfiltered)
# ``locate_rational`` selection drove the certificate's O/X collapse: low-order
# rationals resolve onto the near-axis 1+p/q artifact branch (iota>1) as their
# INNERMOST crossing, where no physical period-q fixed point exists, so Newton stalls.
# Mirrors the real slid_clean trace where locate_rational(1,4) returned iota=1.25.
_NEAR_AXIS_OFFSET_TWIST = (
    (0.010, 1.333),  # near-axis charting offset (iota>1: integer section offset)
    (0.040, 0.250),  # iota falls through the physical band from here out
    (0.060, 0.222),  # physical 2/9 surface
    (0.080, 0.200),  # physical 1/5 surface
    (0.110, 0.167),  # physical 1/6 surface
    (0.140, 0.100),
    (0.170, 0.010),  # approaching the iota->0 separatrix
)


def _profile_from_samples(samples):
    """Build an IotaProfile from (radial_label, iota) pairs (field-independent)."""
    return IotaProfile(
        axis_r=0.8632,
        axis_z=0.0,
        samples=tuple(
            IotaProfileSample(
                radial_label=float(label),
                iota=float(iota),
                toroidal_turns=30,
                min_bphi_over_b=0.5,
                reason="",
            )
            for label, iota in samples
        ),
    )


def test_in_band_selection_excludes_near_axis_charting_offset_artifact():
    """Band-filtered selection keeps physical rationals, drops the iota>1 artifact.

    On a profile whose near-axis ring reads iota>1 (the integer charting offset),
    selecting over the physical band [0, 0.24] must:
      * select the physically realized 2/9, 1/5, 1/6 crossings (their realized
        ``crossing.iota`` lies in band), and
      * NEVER select the near-axis 1/4 *artifact*: raw ``locate_rational(1, 4)``
        returns its INNERMOST crossing, which here is the out-of-band 5/4 branch
        (iota=1.25) -- exactly what the real slid_clean trace produced and what drove
        Newton onto a nonexistent orbit. The band filter screens it out, and the
        physical 1/4 (iota=0.25) sits just above the band, so 1/4 is not probed at all.

    This is the selection-side guard for the O/X collapse: every selected target's
    realized iota lies in the realized physical band, so no probe is ever launched at
    a chart-artifact radius where the period-q orbit does not exist.
    """

    profile = _profile_from_samples(_NEAR_AXIS_OFFSET_TWIST)
    physical_band = (0.0, 0.24)

    # Raw locate_rational(1, 4) resolves onto the out-of-band 5/4 artifact branch
    # (innermost-by-label): the exact defect the band filter must screen out.
    artifact = profile.locate_rational(1, 4)
    assert artifact is not None
    assert artifact.iota == pytest.approx(1.25, abs=1.0e-3)
    assert artifact.iota > physical_band[1]

    selected = _in_band_low_order_crossings(
        profile, iota_band=physical_band, max_denominator=9
    )
    selected_pq = {(c.p, c.q) for c in selected}

    # Every selected crossing's realized iota lies inside the physical band, and is
    # the true low-order rational value (not a shifted n+p/q branch).
    assert len(selected) >= 1
    for crossing in selected:
        assert physical_band[0] - 1.0e-9 <= crossing.iota <= physical_band[1] + 1.0e-9
        assert crossing.iota == pytest.approx(
            crossing.p / crossing.q, abs=1.0e-6
        )

    # Physically realized in-band low-order resonances ARE selected.
    assert (1, 5) in selected_pq  # iota 0.200
    assert (2, 9) in selected_pq  # iota 0.222
    assert (1, 6) in selected_pq  # iota 0.167
    # The 1/4 chain is never probed: its near-axis realization is the out-of-band
    # 1.25 artifact, and its physical value 0.25 sits just above the [0, 0.24] band.
    assert (1, 4) not in selected_pq


def test_residue_iota_metrics_residue_classification_matches_greene_formula():
    """The probe converges on the 1/2 orbit and the module faithfully surfaces R.

    The field is integrable (pure shear, zero island width), so both branches land
    on the marginal trace M ~ 2 (residue ~ 0) fixed point with an area-preserving
    tangent map. The certification-relevant contract here is not the sign of a
    ~1e-10 residue, but that (a) the probe actually converged on the selected
    resonance, (b) the tangent map is symplectic (detM ~ 1), and (c) the
    ``residue_classification`` string the module reports is exactly what
    ``classify_greene_residue`` assigns to the residue it reports -- the module does
    not invent or override the probe's classification.
    """
    metrics = _metrics_on_shear_field()
    aggregate = metrics["aggregate"]

    assert aggregate["n_resonances_probed"] == 1
    # Both O and X branches of the 1/2 target are emitted and converge.
    assert aggregate["n_branch_diagnostics"] == 2
    assert aggregate["n_converged"] == 2
    assert aggregate["branch_status_counts"] == {BRANCH_STATUS_CONVERGED: 2}

    for residue in aggregate["residues"]:
        assert residue["converged"] is True
        assert residue["branch_status"] == BRANCH_STATUS_CONVERGED
        assert residue["winding"] == pytest.approx(1.0, abs=1.0e-6)
        assert residue["detM"] == pytest.approx(1.0, abs=1.0e-5)
        # Integrable field => marginal residue, but the module must report whatever
        # classify_greene_residue says for the residue value it carries.
        assert abs(residue["residue"]) < 1.0e-6
        assert residue["residue_classification"] == classify_greene_residue(
            residue["residue"]
        )


def test_residue_iota_metrics_empty_band_returns_well_formed_zero_aggregate():
    """A field with no in-band low-order crossing yields a clean empty aggregate.

    ``CircularTransformField`` at a constant irrational-looking iota (0.41) realizes
    no reduced p/q with q<=3 inside the (degenerate) band, so nothing is probed.
    The contract: iota is still sampled, no residue probe runs, and the
    confinement boundary reports an unconfined (no converged resonance) verdict
    rather than raising.
    """
    metrics = residue_iota_metrics(
        CircularTransformField(axis_r=1.0, axis_z=0.0, iota=0.41),
        axis_r=1.0,
        axis_z=0.0,
        nfp=1,
        radial_lo=0.1,
        radial_hi=0.3,
        radial_label_count=5,
        max_denominator=3,
        toroidal_turns=8,
        iota_integrator_options=_ANALYTIC_INTEGRATOR,
        integrator_options=_ANALYTIC_INTEGRATOR,
        solver_options=_ANALYTIC_SOLVER,
        phase_angles=(0.0,),
    )

    assert metrics["iota_profile"]["n_valid"] == 5
    assert metrics["rational_crossings"] == []
    assert metrics["residue_probe"] is None
    aggregate = metrics["aggregate"]
    assert aggregate["n_resonances_probed"] == 0
    assert aggregate["n_converged"] == 0
    assert aggregate["residues"] == []
    boundary = metrics["confinement_boundary"]
    assert boundary["confined_core"] is False
    assert boundary["confinement_boundary_radial_label"] is None
    assert boundary["n_converged_resonances"] == 0


# --------------------------------------------------------------------------- #
# Confinement-boundary reduction (STEP 2): a pure function over residue lists.
# Uses the canonical residue values a real elliptic (R=0.5) / hyperbolic (R<0 or
# R>1) orbit yields -- the same values the Greene-residue conventions suite proves.
# --------------------------------------------------------------------------- #
def _residue_entry(p, q, *, residue, label, converged=True, branch="O"):
    """One aggregate['residues']-shaped record (only the fields the reducer reads).

    Defaults to the elliptic O-branch -- the branch whose residue decides island
    confinement. ``branch="X"`` builds the saddle branch the reducer must ignore.
    """
    return {
        "p": p,
        "q": q,
        "branch": branch,
        "converged": converged,
        "residue": residue,
        "radial_label": label,
    }


def test_confinement_boundary_finds_first_intact_to_destroyed_transition():
    """Intact inner chains then destroyed outer chains => boundary at the tear.

    Two intact elliptic chains (R=0.5, 0.3 -- 0<R<1) at labels 0.20, 0.30 then two
    destroyed chains (R=-0.4 hyperbolic, R=1.5 period-doubled past 1) at labels
    0.40, 0.50. The KAM edge is the inner-most destroyed label (0.40), bracketed by
    the 0.30 intact chain.
    """
    residues = [
        _residue_entry(1, 2, residue=0.5, label=0.20),
        _residue_entry(2, 3, residue=0.3, label=0.30),
        _residue_entry(3, 5, residue=-0.4, label=0.40),
        _residue_entry(4, 7, residue=1.5, label=0.50),
    ]

    boundary = confinement_boundary_from_residues(residues)

    assert boundary["confinement_boundary_radial_label"] == pytest.approx(0.40)
    assert boundary["inner_resonance"]["radial_label"] == pytest.approx(0.30)
    assert boundary["inner_resonance"]["residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["outer_resonance"]["radial_label"] == pytest.approx(0.40)
    assert boundary["outer_resonance"]["residue_class"] == RESIDUE_CLASS_DESTROYED
    assert boundary["core_residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["edge_residue_class"] == RESIDUE_CLASS_DESTROYED
    assert boundary["confined_core"] is True
    assert boundary["n_converged_resonances"] == 4


def test_confinement_boundary_classifies_canonical_elliptic_and_hyperbolic_residues():
    """0<R<1 is intact/elliptic; R<0 and R>1 are destroyed/hyperbolic; R==0/1 boundary.

    Anchored on the residue formula's own thresholds (greene_residue.classify):
    an elliptic O-point island has 0<R<1, a hyperbolic X-point chain has R<0 or
    R>1, and the marginal cases (parabolic R=0, period-doubling R=1) are neither --
    reported as the literal boundary class, not silently coerced to confined/torn.
    """
    intact = confinement_boundary_from_residues(
        [_residue_entry(1, 2, residue=0.5, label=0.2)]
    )
    assert classify_greene_residue(0.5) == GREENE_RESIDUE_ELLIPTIC_O
    assert intact["core_residue_class"] == RESIDUE_CLASS_INTACT
    assert intact["confined_core"] is True

    for destroyed_residue in (-0.4, 1.5):
        destroyed = confinement_boundary_from_residues(
            [_residue_entry(1, 2, residue=destroyed_residue, label=0.2)]
        )
        assert classify_greene_residue(destroyed_residue) == GREENE_RESIDUE_HYPERBOLIC_X
        assert destroyed["core_residue_class"] == RESIDUE_CLASS_DESTROYED
        assert destroyed["confined_core"] is False

    for boundary_residue in (0.0, 1.0):
        marginal = confinement_boundary_from_residues(
            [_residue_entry(1, 2, residue=boundary_residue, label=0.2)]
        )
        assert marginal["core_residue_class"] == RESIDUE_CLASS_BOUNDARY
        assert marginal["confined_core"] is False


def test_confinement_boundary_ignores_non_converged_branches():
    """A non-converged branch has no residue to classify, so it cannot place the edge.

    The first (innermost) branch did not converge (residue None); it must be
    dropped, so the core class and the edge are decided by the two converged
    chains: intact at 0.30, destroyed at 0.45 -> boundary 0.45.
    """
    residues = [
        _residue_entry(1, 2, residue=None, label=None, converged=False),
        _residue_entry(2, 3, residue=0.4, label=0.30),
        _residue_entry(3, 5, residue=2.0, label=0.45),
    ]

    boundary = confinement_boundary_from_residues(residues)

    assert boundary["n_converged_resonances"] == 2
    assert boundary["core_residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["confinement_boundary_radial_label"] == pytest.approx(0.45)
    assert boundary["confined_core"] is True


def test_confinement_boundary_uses_o_point_not_hyperbolic_x_saddle():
    """An intact island's own hyperbolic X-saddle must NOT read as a torn chain.

    Every island chain has an elliptic O-point AND a hyperbolic X-point (a saddle,
    R<0) at the same radius. The confinement discriminator is the O-point: a chain
    with 0<R_O<1 is intact regardless of its X-saddle. Here two intact islands each
    contribute a converged O-branch (R=0.5, 0.4) and its hyperbolic X-branch
    (R=-0.6, -0.5) at the same label. The reduction must classify both islands as
    intact from the O-branch and find NO intact->destroyed boundary -- if it counted
    the X-saddles it would manufacture a spurious tear at the inner island's radius.
    """
    residues = [
        _residue_entry(1, 2, residue=0.5, label=0.20, branch=GREENE_BRANCH_O),
        _residue_entry(1, 2, residue=-0.6, label=0.20, branch=GREENE_BRANCH_X),
        _residue_entry(2, 3, residue=0.4, label=0.30, branch=GREENE_BRANCH_O),
        _residue_entry(2, 3, residue=-0.5, label=0.30, branch=GREENE_BRANCH_X),
    ]

    boundary = confinement_boundary_from_residues(residues)

    # Only the two O-branches vote.
    assert boundary["n_converged_resonances"] == 2
    assert boundary["core_residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["edge_residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["confined_core"] is True
    assert boundary["confinement_boundary_radial_label"] is None


def test_confinement_boundary_all_intact_has_confined_core_and_no_edge():
    """An all-intact profile is fully confined: a core, an intact edge, no tear."""
    residues = [
        _residue_entry(1, 2, residue=0.5, label=0.2),
        _residue_entry(2, 3, residue=0.2, label=0.3),
    ]

    boundary = confinement_boundary_from_residues(residues)

    assert boundary["confined_core"] is True
    assert boundary["confinement_boundary_radial_label"] is None
    assert boundary["inner_resonance"] is None
    assert boundary["outer_resonance"] is None
    assert boundary["core_residue_class"] == RESIDUE_CLASS_INTACT
    assert boundary["edge_residue_class"] == RESIDUE_CLASS_INTACT
