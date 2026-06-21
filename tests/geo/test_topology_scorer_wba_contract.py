import math

import numpy as np
import simsopt.field as simsopt_field
from simsopt.field.magnetic_axis_helpers import MagneticAxisNotLocatedError
from simsopt.geo import SurfaceRZFourier

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

import topology_scorer


class RingSurface:
    nfp = 1

    def cross_section(self, *, phi, thetas):
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        return np.column_stack(
            (
                1.1 + 0.1 * np.cos(theta),
                np.zeros_like(theta) + float(phi),
                0.1 * np.sin(theta),
            )
        )


class ShiftedCentroidSurface:
    nfp = 1

    def cross_section(self, *, phi, thetas):
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        return np.column_stack(
            (
                1.22 + 0.14 * np.cos(theta),
                np.zeros_like(theta) + float(phi),
                0.0 + 0.1 * np.sin(theta),
            )
        )


class ShiftedAxisField:
    def __init__(self, *, axis_r, axis_z, iota):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self._points = np.zeros((0, 3), dtype=float)

    def set_points(self, points):
        self._points = np.asarray(points, dtype=float)
        return self

    def B(self):
        vectors = []
        for point in self._points:
            x, y, z = point
            radius = math.hypot(float(x), float(y))
            phi = math.atan2(float(y), float(x))
            e_r = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
            e_phi = np.asarray([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
            e_z = np.asarray([0.0, 0.0, 1.0], dtype=float)

            d_radius_d_phi = -self.iota * (float(z) - self.axis_z)
            d_z_d_phi = self.iota * (radius - self.axis_r)
            b_phi = 1.0
            b_radius = d_radius_d_phi * b_phi / radius
            b_z = d_z_d_phi * b_phi / radius
            vectors.append(b_radius * e_r + b_phi * e_phi + b_z * e_z)
        return np.asarray(vectors, dtype=float)


class BphiNullAxisField(ShiftedAxisField):
    """ShiftedAxisField whose toroidal field vanishes on the cylinder R=r_null.

    The axis at ``(axis_r, axis_z)`` stays healthy, but an off-axis trial field
    line that oscillates across ``r_null`` drives |B_phi|/|B| through zero
    mid-integration -- the RC1 m8n8 condition. Before RC1 this aborted the whole
    axis solve and surfaced as a broken topology score; it must now resolve to a
    broken=False result (axis recovered or gracefully not-located).
    """

    def __init__(self, *, axis_r, axis_z, iota, r_null):
        super().__init__(axis_r=axis_r, axis_z=axis_z, iota=iota)
        self.r_null = float(r_null)

    def B(self):
        vectors = []
        for point in self._points:
            x, y, z = point
            radius = math.hypot(float(x), float(y))
            phi = math.atan2(float(y), float(x))
            e_r = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
            e_phi = np.asarray([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
            e_z = np.asarray([0.0, 0.0, 1.0], dtype=float)

            d_radius_d_phi = -self.iota * (float(z) - self.axis_z)
            d_z_d_phi = self.iota * (radius - self.axis_r)
            b_phi = (radius - self.r_null) / self.axis_r
            b_radius = d_radius_d_phi / radius
            b_z = d_z_d_phi / radius
            vectors.append(b_radius * e_r + b_phi * e_phi + b_z * e_z)
        return np.asarray(vectors, dtype=float)


def test_invariant_torus_classification_reports_not_evaluated_for_short_traces():
    short_hits = np.array(
        [[float(step), 0.0, 1.1, 0.0, 0.0] for step in range(60)],
        dtype=float,
    )

    result = topology_scorer.invariant_torus_classification(
        [short_hits],
        RingSurface(),
        axis_point={"r": 1.1, "z": 0.0, "source": "test_magnetic_axis"},
    )

    assert result["invariant_torus_fraction"] is None
    assert result["wba_classified_seed_count"] == 0
    assert result["wba_classification_counts"]["insufficient_returns"] == 1
    assert result["wba_evaluation_state"] == "not_evaluated_insufficient_returns"


def _golden_invariant_torus_hits(*, axis_r, axis_z, count):
    # A quasi-periodic orbit on an invariant torus (golden rotation number) at a
    # fixed minor radius about the axis -- the WBA classifier resolves it as an
    # invariant_torus once it has >= min_returns points.
    rotation_number = (math.sqrt(5.0) - 1.0) / 10.0
    theta = np.arange(count, dtype=float) * 2.0 * math.pi * rotation_number
    return np.column_stack(
        (
            np.arange(theta.size, dtype=float),
            np.zeros(theta.size, dtype=float),
            axis_r + 0.08 * np.cos(theta),
            np.zeros(theta.size, dtype=float),
            axis_z + 0.08 * np.sin(theta),
        )
    )


def test_wba_min_returns_override_flips_insufficient_to_evaluated():
    # Mirrors the m10n10 in-loop case: an axis-healthy lane whose in-run tmax
    # yields ~130 Poincare returns -- below the strict-validation bar (256) but
    # enough for a coarse in-loop signal. The externally-owned min_returns knob
    # lets a caller classify at the lower bar; the default keeps 256 unchanged.
    axis_r, axis_z = 1.05, 0.16
    hits = _golden_invariant_torus_hits(axis_r=axis_r, axis_z=axis_z, count=130)
    axis_point = {"r": axis_r, "z": axis_z, "source": "test_magnetic_axis"}

    # Two invariant-torus field lines clear the WBA_MIN_CLASSIFIABLE_SEEDS=2 floor
    # (#9 denominator fix), so the fraction is a 2/2 ratio over distinct survivors
    # rather than a degenerate single-seed certification. The min_returns override
    # is what flips both from insufficient_returns to evaluated.
    lowered = topology_scorer.invariant_torus_classification(
        [hits, hits], RingSurface(), axis_point=axis_point, min_returns=128
    )
    assert lowered["wba_evaluation_state"] == "evaluated"
    assert lowered["wba_classified_seed_count"] == 2
    assert lowered["invariant_torus_fraction"] == 1.0
    assert lowered["wba_settings"]["min_returns"] == 128

    # Default (None) keeps the strict 256 bar -> still insufficient on 130 returns.
    default = topology_scorer.invariant_torus_classification(
        [hits, hits], RingSurface(), axis_point=axis_point
    )
    assert default["wba_evaluation_state"] == "not_evaluated_insufficient_returns"
    assert default["invariant_torus_fraction"] is None
    assert (
        default["wba_settings"]["min_returns"]
        == topology_scorer.DEFAULT_BIRKHOFF_CLASSIFIER_SETTINGS.min_returns
        == 256
    )

    # Explicit min_returns=256 is identical to the default (no behavior change).
    explicit_default = topology_scorer.invariant_torus_classification(
        [hits, hits], RingSurface(), axis_point=axis_point, min_returns=256
    )
    assert (
        explicit_default["wba_evaluation_state"]
        == "not_evaluated_insufficient_returns"
    )


def test_wba_settings_known_limitations_reflect_radial_reconciliation():
    # Honest provenance (F-mb2): the persisted wba_settings.known_limitations must
    # describe the layer that produced the fraction. When the scorer ran the
    # radial ω-plateau reconciliation (seed_radial_labels supplied), the published
    # invariant_torus_fraction already incorporates a radial-plateau discriminator,
    # so the emitted limitations must NOT still advertise that it is missing; when
    # reconciliation did not run, the bare single-surface classifier limitation
    # stands. The bare-classifier limitation is owned by kam_birkhoff (SSOT); the
    # scorer composes the effective set here.
    axis_r, axis_z = 1.05, 0.16
    hits = _golden_invariant_torus_hits(axis_r=axis_r, axis_z=axis_z, count=300)
    axis_point = {"r": axis_r, "z": axis_z, "source": "test_magnetic_axis"}
    missing = topology_scorer.WBA_LIMITATION_NO_RADIAL_PLATEAU_DISCRIMINATOR
    marker = topology_scorer.WBA_RADIAL_PLATEAU_RECONCILED_BY_SCORER
    subseed = topology_scorer.RECONCILER_KNOWN_LIMITATION_SUBSEED_SPACING

    # Reconciliation ran: per-seed radial labels supplied for two seeds.
    reconciled = topology_scorer.invariant_torus_classification(
        [hits, hits],
        RingSurface(),
        axis_point=axis_point,
        seed_radial_labels=[axis_r + 0.08, axis_r + 0.10],
    )
    assert reconciled["wba_evaluation_state"] == "evaluated"
    reconciled_limitations = reconciled["wba_settings"]["known_limitations"]
    # Fail-on-old: before the fix the emitted payload still carried the bare
    # missing-radial-discriminator limitation even though the scorer reconciled it.
    assert missing not in reconciled_limitations
    assert marker in reconciled_limitations
    # The discriminator's own residual limitation (it cannot resolve islands
    # narrower than the seed spacing) must be disclosed honestly when it runs.
    assert subseed in reconciled_limitations

    # Reconciliation did NOT run: no radial labels -> bare-classifier limitation
    # stands and the downstream-discriminator marker is absent.
    bare = topology_scorer.invariant_torus_classification(
        [hits, hits],
        RingSurface(),
        axis_point=axis_point,
        seed_radial_labels=None,
    )
    assert bare["wba_evaluation_state"] == "evaluated"
    bare_limitations = bare["wba_settings"]["known_limitations"]
    assert missing in bare_limitations
    assert marker not in bare_limitations
    # The reconciliation did not run, so its resolution limitation is not published.
    assert subseed not in bare_limitations


def test_score_topology_threads_wba_min_returns_into_classifier(monkeypatch):
    # The score_topology -> invariant_torus_classification wiring must forward
    # wba_min_returns so the in-loop scorer pass can use the cheaper bar.
    captured = {}

    def fake_itc(*_args, **kwargs):
        captured["min_returns"] = kwargs.get("min_returns")
        payload = topology_scorer.wba_not_evaluated_payload(
            topology_scorer.WBA_EVALUATION_NOT_EVALUATED_SKIPPED_BY_CALLER
        )
        payload["wba_min_evaluable_share"] = 0.7
        payload["wba_evaluable_share"] = 0.25
        return payload

    monkeypatch.setattr(
        topology_scorer,
        "build_stopping_criteria",
        lambda *_args, **_kwargs: ([object()], ["surface_exit"]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "midplane_seed_radii",
        lambda *_args, **_kwargs: np.array([1.0, 1.1]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "prepare_topology_field",
        lambda *_args, **_kwargs: (object(), {"selected_mode": "native"}),
    )
    monkeypatch.setattr(topology_scorer, "cross_section_span", lambda *_args: 1.0)
    monkeypatch.setattr(topology_scorer, "invariant_torus_classification", fake_itc)
    monkeypatch.setattr(
        simsopt_field,
        "compute_fieldlines",
        lambda *_args, **_kwargs: (
            [np.array([[0.0, 1.0, 0.0, 0.0]])],
            [np.array([[0.4, 0.0, 1.0, 0.0, 0.0]])],
        ),
    )

    result = topology_scorer.score_topology(
        RingSurface(),
        object(),
        nfieldlines=2,
        tmax=2.0,
        tol=1e-7,
        nphis=1,
        field_policy="never",
        compute_transport_diagnostics=False,
        compute_invariant_torus_classification=True,
        wba_min_returns=128,
    )

    assert captured["min_returns"] == 128
    assert result["wba_min_evaluable_share"] == 0.7
    assert result["wba_evaluable_share"] == 0.25


def test_score_topology_axis_bphi_null_trial_is_not_broken(monkeypatch):
    # RC1: a field whose axis solve hits a transient B_phi null on an off-axis
    # trial step must NOT poison the topology score as broken=True. The real
    # poloidal_axis_point runs here (not monkeypatched); the residual absorbs the
    # null so the solve recovers the healthy axis (or ends gracefully), and the
    # finalized score is broken=False with no escaping evaluation error.
    fieldlines_tys = [
        np.array([[0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.1, 0.0, 0.0]]),
    ]
    fieldlines_phi_hits = [
        np.array([[0.4, 0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 1.1, 0.0, 0.0]]),
    ]

    monkeypatch.setattr(
        topology_scorer,
        "build_stopping_criteria",
        lambda *_args, **_kwargs: ([object()], ["surface_exit"]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "midplane_seed_radii",
        lambda *_args, **_kwargs: np.array([1.0, 1.1]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "prepare_topology_field",
        lambda *_args, **_kwargs: (object(), {"selected_mode": "native"}),
    )
    monkeypatch.setattr(topology_scorer, "cross_section_span", lambda *_args: 1.0)
    monkeypatch.setattr(
        simsopt_field,
        "compute_fieldlines",
        lambda *_args, **_kwargs: (fieldlines_tys, fieldlines_phi_hits),
    )

    # RingSurface centroid is ~ (1.1, 0); axis at 1.1, B_phi null cylinder at 0.97
    # sits inside the seed-bounds span so a trial excursion crosses it.
    bfield = BphiNullAxisField(axis_r=1.1, axis_z=0.0, iota=0.3, r_null=0.97)

    result = topology_scorer.finalize_topology_score_result(
        topology_scorer.score_topology(
            RingSurface(),
            bfield,
            nfieldlines=2,
            tmax=2.0,
            tol=1e-7,
            nphis=1,
            field_policy="never",
            compute_transport_diagnostics=False,
            compute_invariant_torus_classification=True,
        )
    )

    assert result["broken"] is False
    assert result["evaluation_state"] == "evaluated"
    assert result["evaluation_error"] is None


def test_wba_reports_not_evaluated_without_magnetic_axis_reference():
    short_hits = np.array(
        [[float(step), 0.0, 1.1, 0.0, 0.0] for step in range(60)],
        dtype=float,
    )

    result = topology_scorer.invariant_torus_classification(
        [short_hits],
        RingSurface(),
    )

    assert result["invariant_torus_fraction"] is None
    assert result["wba_classified_seed_count"] == 0
    assert result["wba_axis"] is None
    assert result["wba_classification_counts"]["missing_magnetic_axis"] == 1
    assert result["wba_evaluation_state"] == "not_evaluated_missing_magnetic_axis"


def test_wba_uses_magnetic_axis_instead_of_boundary_centroid():
    axis_r = 1.05
    axis_z = 0.16
    rotation_number = (math.sqrt(5.0) - 1.0) / 10.0
    theta = np.arange(300, dtype=float) * 2.0 * math.pi * rotation_number
    radius = axis_r + 0.08 * np.cos(theta)
    z = axis_z + 0.08 * np.sin(theta)
    hits = np.column_stack(
        (
            np.arange(theta.size, dtype=float),
            np.zeros(theta.size, dtype=float),
            radius,
            np.zeros(theta.size, dtype=float),
            z,
        )
    )

    # Two identical invariant-torus field lines clear the
    # WBA_MIN_CLASSIFIABLE_SEEDS=2 floor (#9 denominator fix); both share the same
    # off-centroid magnetic axis, so the axis-vs-centroid intent is unchanged
    # while the fraction is now a 2/2 ratio over distinct survivors.
    result = topology_scorer.invariant_torus_classification(
        [hits, hits],
        ShiftedCentroidSurface(),
        bfield=ShiftedAxisField(axis_r=axis_r, axis_z=axis_z, iota=rotation_number),
    )

    surface_centroid = topology_scorer._surface_phi0_bounds(ShiftedCentroidSurface())[2]
    assert result["wba_axis"]["source"] == "magnetic_axis_fieldline_fixed_point"
    assert abs(result["wba_axis"]["r"] - axis_r) < 1.0e-6
    assert abs(result["wba_axis"]["z"] - axis_z) < 1.0e-6
    assert abs(result["wba_axis"]["r"] - surface_centroid[0]) > 1.0e-2
    assert abs(result["wba_axis"]["z"] - surface_centroid[1]) > 1.0e-2
    assert result["wba_evaluation_state"] == "evaluated"
    assert result["wba_classified_seed_count"] == 2
    assert result["invariant_torus_fraction"] == 1.0


def test_empty_topology_score_does_not_promote_not_evaluated_wba_to_kam_fraction():
    result = topology_scorer.empty_topology_score_result(12, 50.0)

    assert result["invariant_torus_fraction"] is None
    assert result["kam_fraction"] is None
    assert result["kam_fraction_semantics"] is None
    assert result["wba_evaluation_state"] == "not_evaluated_no_classified_seeds"
    assert (
        result["wba_min_evaluable_share"]
        == topology_scorer.WBA_MIN_EVALUABLE_SHARE
    )
    assert result["wba_evaluable_share"] == 0.0


def test_score_topology_axis_not_located_is_wba_not_evaluated(monkeypatch):
    fieldlines_tys = [
        np.array([[0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.1, 0.0, 0.0]]),
    ]
    fieldlines_phi_hits = [
        np.array([[0.4, 0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 1.1, 0.0, 0.0]]),
    ]
    stop_labels = [
        "surface_exit",
        "max_z_guardrail",
        "min_z_guardrail",
        "min_r_guardrail",
        "max_r_guardrail",
        "iteration_limit",
    ]

    monkeypatch.setattr(
        topology_scorer,
        "build_stopping_criteria",
        lambda *_args, **_kwargs: ([object()], stop_labels),
    )
    monkeypatch.setattr(
        topology_scorer,
        "midplane_seed_radii",
        lambda *_args, **_kwargs: np.array([1.0, 1.1]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "prepare_topology_field",
        lambda *_args, **_kwargs: (object(), {"selected_mode": "native"}),
    )
    monkeypatch.setattr(topology_scorer, "cross_section_span", lambda *_args: 1.0)
    monkeypatch.setattr(
        topology_scorer,
        "poloidal_axis_point",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MagneticAxisNotLocatedError("axis not located")
        ),
    )
    monkeypatch.setattr(
        simsopt_field,
        "compute_fieldlines",
        lambda *_args, **_kwargs: (fieldlines_tys, fieldlines_phi_hits),
    )

    result = topology_scorer.finalize_topology_score_result(
        topology_scorer.score_topology(
            RingSurface(),
            object(),
            nfieldlines=2,
            tmax=2.0,
            tol=1e-7,
            nphis=1,
            field_policy="never",
            compute_transport_diagnostics=False,
            compute_invariant_torus_classification=True,
        )
    )

    assert result["broken"] is False
    assert result["evaluation_state"] == "evaluated"
    assert result["evaluation_error"] is None
    assert result["wba_evaluation_state"] == "not_evaluated_axis_not_located"
    assert result["wba_not_evaluated_reason"] == "not_evaluated_axis_not_located"
    assert result["invariant_torus_fraction"] is None
    assert result["kam_fraction"] is None


def test_score_topology_can_skip_wba_axis_classification(monkeypatch):
    fieldlines_tys = [
        np.array([[0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.1, 0.0, 0.0]]),
    ]
    fieldlines_phi_hits = [
        np.array([[0.4, 0.0, 1.0, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 1.1, 0.0, 0.0]]),
    ]
    stop_labels = [
        "surface_exit",
        "max_z_guardrail",
        "min_z_guardrail",
        "min_r_guardrail",
        "max_r_guardrail",
        "iteration_limit",
    ]

    monkeypatch.setattr(
        topology_scorer,
        "build_stopping_criteria",
        lambda *_args, **_kwargs: ([object()], stop_labels),
    )
    monkeypatch.setattr(
        topology_scorer,
        "midplane_seed_radii",
        lambda *_args, **_kwargs: np.array([1.0, 1.1]),
    )
    monkeypatch.setattr(
        topology_scorer,
        "prepare_topology_field",
        lambda *_args, **_kwargs: (object(), {"selected_mode": "native"}),
    )
    monkeypatch.setattr(topology_scorer, "cross_section_span", lambda *_args: 1.0)
    monkeypatch.setattr(
        topology_scorer,
        "invariant_torus_classification",
        _raise_unexpected_wba_call,
    )
    monkeypatch.setattr(
        simsopt_field,
        "compute_fieldlines",
        lambda *_args, **_kwargs: (fieldlines_tys, fieldlines_phi_hits),
    )

    result = topology_scorer.score_topology(
        RingSurface(),
        object(),
        nfieldlines=2,
        tmax=2.0,
        tol=1e-7,
        nphis=1,
        field_policy="never",
        compute_transport_diagnostics=False,
        compute_invariant_torus_classification=False,
    )

    assert result["wba_evaluation_state"] == "not_evaluated_skipped_by_caller"
    assert result["wba_not_evaluated_reason"] == "not_evaluated_skipped_by_caller"
    assert result["invariant_torus_fraction"] is None
    assert result["kam_fraction"] is None


def _raise_unexpected_wba_call(*_args, **_kwargs):
    raise AssertionError("search gate WBA path should not run")


def test_score_topology_solves_axis_on_exact_field_under_interpolation(monkeypatch):
    # Long confinement traces run on the InterpolatedField for speed, but the
    # magnetic-axis fixed-point solve must use the EXACT field: an interpolated
    # field cannot close the return map to the 1e-8 axis tolerance, so routing
    # the interpolated tracer into the axis solve made the WBA/topology score
    # report "broken" on valid configs whenever interpolation turned on
    # (tmax >= threshold). This guards score_topology against that regression.
    major_radius = 1.0
    field = simsopt_field.ToroidalField(
        major_radius, 1.0
    ) + simsopt_field.PoloidalField(major_radius, 1.0, 3.0)
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 32, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 32, endpoint=False),
    )
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)

    captured = {}
    real_compute_fieldlines = simsopt_field.compute_fieldlines
    real_classify = topology_scorer.invariant_torus_classification

    def spy_compute_fieldlines(traced_field, *args, **kwargs):
        captured["traced_field"] = traced_field
        return real_compute_fieldlines(traced_field, *args, **kwargs)

    def spy_classify(
        fieldlines_phi_hits,
        surface_arg,
        *,
        bfield=None,
        axis_point=None,
        min_returns=None,
        seed_radial_labels=None,
    ):
        captured["axis_field"] = bfield
        captured["seed_radial_labels"] = seed_radial_labels
        return real_classify(
            fieldlines_phi_hits,
            surface_arg,
            bfield=bfield,
            axis_point=axis_point,
            min_returns=min_returns,
            seed_radial_labels=seed_radial_labels,
        )

    monkeypatch.setattr(simsopt_field, "compute_fieldlines", spy_compute_fieldlines)
    monkeypatch.setattr(topology_scorer, "invariant_torus_classification", spy_classify)

    result = topology_scorer.score_topology(
        surface,
        field,
        nfieldlines=3,
        tmax=60.0,
        nphis=4,
        interpolation_grid={"nr": 8, "nphi": 4, "nz": 8, "degree": 2},
        compute_transport_diagnostics=False,
    )

    interpolated_field_type = simsopt_field.InterpolatedField
    # Interpolation was active: the long traces ran on the interpolated field.
    assert isinstance(captured["traced_field"], interpolated_field_type)
    # ...but the axis solve received the exact field, not the interpolated tracer.
    assert captured["axis_field"] is field
    assert not isinstance(captured["axis_field"], interpolated_field_type)
    # The axis was located at exact-field precision.
    assert result["wba_axis"] is not None
    assert result["wba_axis"]["normalized_return_residual"] < 1.0e-8
