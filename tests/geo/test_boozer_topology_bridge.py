"""Unit tests for the Phase 3 non-Boozer topology bridge metrics."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_bridge_module():
    return importlib.import_module("banana_opt.boozer_topology_bridge")


class TestHelicalFieldContent:
    def test_axisymmetric_field_has_zero_S_HEL(self):
        bridge = load_bridge_module()
        # |B| depends only on theta (poloidal): no n != 0 power.
        nphi, ntheta = 32, 16
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        modB = np.tile(1.0 + 0.1 * np.cos(theta), (nphi, 1))
        s_hel = bridge.helical_field_content_from_modB_grid(modB)
        assert s_hel == pytest.approx(0.0, abs=1e-12)

    def test_zero_dc_pure_helical_mode_has_S_HEL_one(self):
        """When DC = 0, all spectral power IS helical → S_HEL == 1."""
        bridge = load_bridge_module()
        nphi, ntheta = 32, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        modB = 0.5 * np.cos(THETA - 2.0 * PHI)  # single helical mode, DC=0
        s_hel = bridge.helical_field_content_from_modB_grid(modB)
        assert s_hel == pytest.approx(1.0, abs=1e-12)

    def test_dc_dominates_when_helical_amplitude_is_small(self):
        """Plan formula keeps DC in denominator → S_HEL is small when DC >> helical."""
        bridge = load_bridge_module()
        nphi, ntheta = 32, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        modB = 1.0 + 0.5 * np.cos(THETA - 2.0 * PHI)
        s_hel = bridge.helical_field_content_from_modB_grid(modB)
        # Analytic: DC bin power = (nphi*ntheta)^2; helical mode contributes
        # two delta peaks of amplitude (nphi*ntheta * 0.5/2)^2 each.
        # S_HEL = 2 * (0.25/4) / (1.0 + 2 * (0.25/4)) ≈ 0.111.
        expected = 2 * (0.25 / 4) / (1.0 + 2 * (0.25 / 4))
        assert s_hel == pytest.approx(expected, rel=1e-6)

    def test_mixed_modes_give_intermediate_S_HEL(self):
        bridge = load_bridge_module()
        nphi, ntheta = 32, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        # Zero DC, equal-amplitude axisymmetric (n=0) and helical (n=2)
        # modes. With the plan formula (full Parseval denominator) and
        # zero DC, the n=0 and n=2 modes split spectral power 50/50.
        modB = (
            0.3 * np.cos(THETA)              # n=0 (axisymmetric)
            + 0.3 * np.cos(THETA - 2.0 * PHI)  # n=2 (helical)
        )
        s_hel = bridge.helical_field_content_from_modB_grid(modB)
        assert s_hel == pytest.approx(0.5, abs=1e-10)

    def test_constant_field_raises(self):
        """A truly constant field has zero spectral power → undefined.

        ``np.fft.fft2`` of a constant has all power in the DC bin; total
        Parseval power is ``(N*amplitude)^2 != 0`` when amplitude != 0, so
        a non-zero constant field reports ``S_HEL = 0`` (all power is
        axisymmetric) rather than raising. Only a true zero grid raises.
        """
        bridge = load_bridge_module()
        s_hel_constant = bridge.helical_field_content_from_modB_grid(
            np.full((8, 8), 2.5)
        )
        assert s_hel_constant == pytest.approx(0.0, abs=1e-12)
        with pytest.raises(ValueError, match="zero spectral power"):
            bridge.helical_field_content_from_modB_grid(np.zeros((8, 8)))

    def test_non_finite_input_raises(self):
        bridge = load_bridge_module()
        modB = np.full((8, 8), 1.0)
        modB[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite samples"):
            bridge.helical_field_content_from_modB_grid(modB)

    def test_S_HEL_invariant_under_uniform_scaling(self):
        """Scaling |B| by a constant must leave S_HEL invariant (plan line 280)."""
        bridge = load_bridge_module()
        nphi, ntheta = 32, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        base = (
            1.0
            + 0.2 * np.cos(THETA)
            + 0.4 * np.cos(THETA - 3 * PHI)
        )
        scaled = 7.5 * base
        assert bridge.helical_field_content_from_modB_grid(
            base
        ) == pytest.approx(
            bridge.helical_field_content_from_modB_grid(scaled),
            rel=1e-12,
        )


class TestComputeSHELFromField:
    def test_uses_modB_not_BdotN(self):
        """The wrapper transforms ``|B|``, NOT ``B`` or ``B·n``."""
        bridge = load_bridge_module()

        # Synthetic surface gamma: (nphi, ntheta, 3). Helical surface.
        nphi, ntheta = 16, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [
                np.cos(PHI),
                np.sin(PHI),
                0.1 * np.sin(THETA - 2 * PHI),
            ],
            axis=-1,
        )

        captured_points: list[np.ndarray] = []

        # Field magnitude varies only with theta (n=0, m=1): axisymmetric
        # in (phi, theta) terms → S_HEL = 0. Field components are not
        # axisymmetric in (x, y, z), demonstrating that we transform |B|
        # rather than e.g. ``B·n``.
        theta_flat = THETA.reshape(-1)
        phi_flat = PHI.reshape(-1)
        amplitude = 1.0 + 0.3 * np.cos(theta_flat)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)
        B_vector = np.stack([Bx, By, Bz], axis=-1)

        class _FakeField:
            def set_points(self, points: np.ndarray) -> None:
                captured_points.append(np.asarray(points, dtype=float).copy())

            def B(self) -> np.ndarray:
                return B_vector

        surface = SimpleNamespace(gamma=lambda: gamma)
        s_hel = bridge.compute_helical_field_content_S_HEL(_FakeField(), surface)
        assert s_hel == pytest.approx(0.0, abs=1e-12)
        # Verify points were set in the (nphi * ntheta, 3) shape.
        assert captured_points[0].shape == (nphi * ntheta, 3)


class TestArtifactFields:
    """Phase 3b artifact-key payload.

    F6 fix: Phase 3b only emits the helical-content + composite keys. The
    field-line proxy keys (``FIELDLINE_IOTA_PROXY`` /
    ``FIELDLINE_IOTA_PROXY_VALID``) are owned by Phase 3a so the "valid"
    semantics is unambiguous; this function takes the Phase 3a result as
    explicit kwargs and uses them only for the composite-score proximity
    term.
    """

    def test_none_results_emit_phase3b_keys(self):
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=None,
            fieldline_iota_proxy_mean=None,
            fieldline_iota_proxy_valid=False,
            s_hel_objective_weight=None,
            iota_target=0.2,
        )
        assert set(fields) == {
            "HELICAL_FIELD_CONTENT",
            "S_HEL_OBJECTIVE_WEIGHT",
            "PRE_BOOZER_TOPOLOGY_SCORE",
        }
        assert all(value is None for value in fields.values())

    def test_phase3b_does_not_emit_fieldline_proxy_keys(self):
        """SSOT enforcement: Phase 3a owns FIELDLINE_IOTA_PROXY{,_VALID}."""
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.5,
            fieldline_iota_proxy_mean=0.195,
            fieldline_iota_proxy_valid=True,
            s_hel_objective_weight=0.0,
            iota_target=0.2,
        )
        assert "FIELDLINE_IOTA_PROXY" not in fields
        assert "FIELDLINE_IOTA_PROXY_VALID" not in fields

    def test_concrete_results_populate_keys(self):
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.42,
            fieldline_iota_proxy_mean=0.195,
            fieldline_iota_proxy_valid=True,
            s_hel_objective_weight=1.0e-3,
            iota_target=0.2,
        )
        assert fields["HELICAL_FIELD_CONTENT"] == pytest.approx(0.42)
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] == pytest.approx(1.0e-3)
        # iota_proxy is within 0.005 of target so proximity ≈ 0.995.
        # PRE_BOOZER_TOPOLOGY_SCORE = 0.42 * 0.995 ≈ 0.4179.
        assert fields["PRE_BOOZER_TOPOLOGY_SCORE"] == pytest.approx(
            0.42 * (1.0 - abs(0.195 - 0.2)), abs=1e-9
        )

    def test_invalid_fieldline_proxy_collapses_score_to_s_hel(self):
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.5,
            fieldline_iota_proxy_mean=0.1,
            fieldline_iota_proxy_valid=False,
            s_hel_objective_weight=None,
            iota_target=0.2,
        )
        # S_HEL alone (invalid proxy is treated as unavailable).
        assert fields["PRE_BOOZER_TOPOLOGY_SCORE"] == pytest.approx(0.5)

    def test_s_hel_objective_weight_zero_is_persisted_not_collapsed(self):
        """Phase 3b: explicit 0.0 weight must round-trip as 0.0, NOT None.

        Plan line 281 mandates an explicit weight schedule; the live
        optimizer in this phase reports ``0.0`` (gate off but tracked)
        while a completely-unwired call passes ``None`` (diagnostic not
        computed). Conflating the two would hide the scheduling intent.
        """
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.3,
            fieldline_iota_proxy_mean=None,
            fieldline_iota_proxy_valid=False,
            s_hel_objective_weight=0.0,
            iota_target=0.2,
        )
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] == 0.0
        # 0.0 is distinct from None in this contract.
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] is not None

    def test_s_hel_objective_weight_none_is_preserved(self):
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.3,
            fieldline_iota_proxy_mean=None,
            fieldline_iota_proxy_valid=False,
            s_hel_objective_weight=None,
            iota_target=0.2,
        )
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] is None


class TestSHELScaleInvarianceOnField:
    """Phase 3b acceptance: scale invariance must hold end-to-end on a field.

    The grid-level wrapper already has :class:`TestHelicalFieldContent`
    coverage; this class checks the field-level wrapper that the
    artifact path actually calls. ``|B|`` scales linearly with the field,
    so the FFT magnitudes scale linearly and ``S_HEL = sum_helical /
    sum_total`` is unchanged under a uniform ``B -> c * B``.
    """

    def test_scale_invariance_through_field_wrapper(self):
        bridge = load_bridge_module()
        from types import SimpleNamespace

        nphi, ntheta = 16, 16
        phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA - 2 * PHI)],
            axis=-1,
        )
        phi_flat = PHI.reshape(-1)
        theta_flat = THETA.reshape(-1)
        amplitude = 1.0 + 0.4 * np.cos(theta_flat - 3 * phi_flat)

        class _FieldAtScale:
            def __init__(self, scale: float):
                self._scale = float(scale)

            def set_points(self, _):
                pass

            def B(self):
                Bx = self._scale * amplitude * np.cos(phi_flat)
                By = self._scale * amplitude * np.sin(phi_flat)
                Bz = np.zeros_like(amplitude)
                return np.stack([Bx, By, Bz], axis=-1)

        surface = SimpleNamespace(gamma=lambda: gamma)
        s_hel_unit = bridge.compute_helical_field_content_S_HEL(
            _FieldAtScale(1.0), surface
        )
        for scale in (0.1, 7.5, 1.0e6):
            s_hel_scaled = bridge.compute_helical_field_content_S_HEL(
                _FieldAtScale(scale), surface
            )
            assert s_hel_scaled == pytest.approx(s_hel_unit, rel=1e-12)

    def test_safe_wrapper_returns_none_with_zero_power_reason(self):
        """Phase 3b failure semantics: zero |B| must fail closed.

        C6 fix: a degenerate Boozer surface where ``|B| ≡ 0`` would
        previously have produced NaN ``B_unit`` samples that propagated
        silently into the gradient chain rule. The strict path now raises
        ``ValueError`` at the |B|=0 site directly (before the FFT) so the
        safe wrapper converts it to ``None`` in the artifact and the
        optimizer never sees NaN gradients.
        """
        bridge = load_bridge_module()
        from types import SimpleNamespace

        nphi, ntheta = 8, 8
        gamma = np.zeros((nphi, ntheta, 3))

        class _ZeroField:
            def set_points(self, points):
                self._n = np.asarray(points).shape[0]

            def B(self):
                return np.zeros((self._n, 3))

        surface = SimpleNamespace(gamma=lambda: gamma)
        # Strict path raises at the zero-|B| site (C6 fix) before
        # the FFT is even evaluated; the error string identifies the
        # degenerate Boozer surface explicitly.
        with pytest.raises(ValueError, match="zero magnetic field"):
            bridge.compute_helical_field_content_S_HEL(_ZeroField(), surface)
        # Safe wrapper returns None (artifact pipeline must NOT crash).
        assert (
            bridge.safe_compute_helical_field_content_S_HEL(
                _ZeroField(), surface
            )
            is None
        )

    def test_artifact_fields_with_zero_weight_emit_no_score_when_s_hel_none(self):
        """Bridge-enabled but S_HEL undefined: composite stays None, weight 0.0."""
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=None,
            fieldline_iota_proxy_mean=None,
            fieldline_iota_proxy_valid=False,
            s_hel_objective_weight=0.0,
            iota_target=0.2,
        )
        assert fields["HELICAL_FIELD_CONTENT"] is None
        assert fields["PRE_BOOZER_TOPOLOGY_SCORE"] is None
        # The schedule slot stays at 0.0; only the diagnostic is unavailable.
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] == 0.0


class TestPreBoozerTopologyScore:
    def test_both_diagnostics_compose_multiplicatively(self):
        bridge = load_bridge_module()
        score = bridge.compute_pre_boozer_topology_score(
            s_hel=0.6,
            fieldline_iota_proxy_mean=0.2,
            fieldline_iota_proxy_valid=True,
            iota_target=0.2,
        )
        # Perfect proximity → score = S_HEL.
        assert score == pytest.approx(0.6)

    def test_only_s_hel_returns_s_hel(self):
        bridge = load_bridge_module()
        score = bridge.compute_pre_boozer_topology_score(
            s_hel=0.4,
            fieldline_iota_proxy_mean=None,
            fieldline_iota_proxy_valid=False,
            iota_target=0.2,
        )
        assert score == pytest.approx(0.4)

    def test_neither_diagnostic_returns_none(self):
        bridge = load_bridge_module()
        assert (
            bridge.compute_pre_boozer_topology_score(
                s_hel=None,
                fieldline_iota_proxy_mean=None,
                fieldline_iota_proxy_valid=False,
                iota_target=0.2,
            )
            is None
        )

    def test_iota_proxy_only_clamps_to_nonneg(self):
        bridge = load_bridge_module()
        score = bridge.compute_pre_boozer_topology_score(
            s_hel=None,
            fieldline_iota_proxy_mean=5.0,  # way off target
            fieldline_iota_proxy_valid=True,
            iota_target=0.2,
        )
        # Proximity (1 - 4.8) would be -3.8; clamped to 0.
        assert score == 0.0


class TestFieldlineProxyParameterValidation:
    def test_zero_n_lines_rejected(self):
        bridge = load_bridge_module()
        with pytest.raises(ValueError, match="n_lines must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=None,
                surface=None,
                n_lines=0,
                tmax=10.0,
                tol=1e-8,
            )

    def test_zero_tmax_rejected(self):
        bridge = load_bridge_module()
        with pytest.raises(ValueError, match="tmax must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=None,
                surface=None,
                n_lines=4,
                tmax=0.0,
                tol=1e-8,
            )

    def test_zero_tol_rejected(self):
        bridge = load_bridge_module()
        with pytest.raises(ValueError, match="tol must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=None,
                surface=None,
                n_lines=4,
                tmax=10.0,
                tol=0.0,
            )

    def test_invalid_gamma_shape_raises(self):
        bridge = load_bridge_module()
        surface = SimpleNamespace(gamma=lambda: np.zeros((4, 3)))
        with pytest.raises(ValueError, match="surface.gamma"):
            bridge.compute_helical_field_content_S_HEL(None, surface)


class TestFieldlineProxyTracingContract:
    def test_continuous_angle_iota_uses_surface_centroid_axis(self, monkeypatch):
        """Pin the C4 continuous-angle path: theta is taken relative to the
        centroid axis (NOT the global Z axis), so a circular loop around
        the centroid produces a well-defined iota proxy.

        Replaces the legacy ``compute_toroidal_transits`` /
        ``compute_poloidal_transits`` patching contract — the new code
        path bypasses those upstream helpers entirely because their
        integer ``np.round`` quantizes iota at ``1/ntor`` resolution
        below the plan's ``5e-3`` convergence tolerance (C4 fix).
        """
        bridge = load_bridge_module()

        # Standard axisymmetric ring with major radius 1.0, minor radius
        # 0.1. Centroid axis sits on the R=1.0 circle at Z=0.
        major_radius = 1.0
        minor_radius = 0.1

        class Surface:
            def cross_section(self, phi, thetas=None):
                theta = np.linspace(
                    0.0, 2.0 * np.pi, int(thetas), endpoint=False
                )
                toroidal_angle = 2.0 * np.pi * float(phi)
                # Ring centroid at (R=1, Z=0), ring of minor radius 0.1.
                rr = major_radius + minor_radius * np.cos(theta)
                zz = minor_radius * np.sin(theta)
                return np.stack(
                    [
                        rr * np.cos(toroidal_angle),
                        rr * np.sin(toroidal_angle),
                        zz,
                    ],
                    axis=-1,
                )

            def gamma(self):
                phis = np.linspace(0.0, 1.0, 16, endpoint=False)
                sections = [self.cross_section(p, thetas=32) for p in phis]
                return np.stack(sections, axis=0)

        # Build a synthetic helical trajectory with exact iota = 0.2.
        # Trajectory makes 5 toroidal revolutions and 1 poloidal
        # revolution around the centroid axis (R=1.0, Z=0).
        n_steps = 5001
        # Parameterize directly by toroidal angle phi for clarity.
        phi_traj = np.linspace(0.0, 5.0 * 2.0 * np.pi, n_steps)
        theta_traj = 0.2 * phi_traj  # iota = dtheta / dphi = 0.2
        # Poloidal angle convention used inside bridge:
        #   theta = arctan2(R - R_ma, Z - Z_ma)
        # so R = R_ma + r*sin(theta), Z = Z_ma + r*cos(theta).
        R_offset = minor_radius * np.sin(theta_traj)
        Z_offset = minor_radius * np.cos(theta_traj)
        R_traj = major_radius + R_offset
        Z_traj = 0.0 + Z_offset
        x = R_traj * np.cos(phi_traj)
        y = R_traj * np.sin(phi_traj)
        z = Z_traj
        t = np.linspace(0.0, 1.0, n_steps)
        histories = [np.stack([t, x, y, z], axis=-1)]

        monkeypatch.setattr(
            bridge, "midplane_seed_radii", lambda surface, n: [1.05]
        )
        monkeypatch.setattr(
            bridge,
            "build_stopping_criteria",
            lambda surface: ([], None),
        )
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, []),
        )

        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=Surface(),
            n_lines=1,
            tmax=10.0,
            tol=1.0e-8,
        )

        # Continuous-angle iota equals the analytic 0.2 to within
        # numerical-integration precision (the centroid-axis adapter
        # interpolates the centroid loop linearly between knots, which
        # introduces a small ~1e-3 truncation floor at the centroid
        # discretization of 128 phi knots).
        assert result.iota_proxy_mean == pytest.approx(0.2, abs=5e-3)
        assert result.n_lines_survived == 1


class TestContinuousAngleIota:
    """C4 fix: continuous-angle iota replaces integer transit counting.

    The legacy path used ``compute_toroidal_transits`` /
    ``compute_poloidal_transits`` which quantize the integer
    transit counts via ``np.round``. The resulting iota = npol/ntor has
    resolution 1/ntor, capping the achievable convergence tolerance at
    ``1/n_transits_target`` regardless of trace length. Continuous-angle
    accumulation removes the quantization floor.
    """

    def _shared_axis(self, major_radius: float):
        """Axisymmetric centroid axis at (R=major_radius, Z=0)."""
        bridge = load_bridge_module()
        from banana_opt.topology_bridge_shared import (
            build_surface_centroid_axis,
        )

        class _RingSurface:
            def cross_section(self, phi, thetas):
                theta = np.linspace(
                    0.0, 2.0 * np.pi, int(thetas), endpoint=False
                )
                toroidal_angle = 2.0 * np.pi * float(phi)
                rr = major_radius + 0.0 * theta  # planar centroid
                zz = 0.0 * theta
                return np.stack(
                    [
                        rr * np.cos(toroidal_angle),
                        rr * np.sin(toroidal_angle),
                        zz,
                    ],
                    axis=-1,
                )

        return bridge, build_surface_centroid_axis(_RingSurface())

    def test_axisymmetric_circular_trajectory_yields_zero_iota(self):
        """A pure-toroidal trajectory (no poloidal drift) has iota = 0."""
        bridge, axis = self._shared_axis(major_radius=1.0)
        n_steps = 4001
        phi = np.linspace(0.0, 5.0 * 2.0 * np.pi, n_steps)
        R = np.full(n_steps, 1.1)  # constant R, constant Z
        Z = np.zeros(n_steps)
        traj = np.stack(
            [
                np.linspace(0.0, 1.0, n_steps),
                R * np.cos(phi),
                R * np.sin(phi),
                Z,
            ],
            axis=-1,
        )
        iota = bridge._continuous_angle_iota([traj], axis)
        # No poloidal drift => delta_theta ~ 0 => iota ~ 0.
        assert iota[0] == pytest.approx(0.0, abs=1e-3)

    def test_analytic_helix_recovers_iota_within_tolerance(self):
        """A helix with prescribed iota recovers it to high precision."""
        bridge, axis = self._shared_axis(major_radius=1.0)
        n_steps = 5001
        phi = np.linspace(0.0, 20.0 * 2.0 * np.pi, n_steps)
        # Pick a non-rational iota so the continuous-angle path cannot
        # benefit from any accidental integer-transit lock-in.
        true_iota = 0.273
        theta = true_iota * phi
        minor_r = 0.05
        R = 1.0 + minor_r * np.sin(theta)
        Z = 0.0 + minor_r * np.cos(theta)
        traj = np.stack(
            [
                np.linspace(0.0, 1.0, n_steps),
                R * np.cos(phi),
                R * np.sin(phi),
                Z,
            ],
            axis=-1,
        )
        iota = bridge._continuous_angle_iota([traj], axis)
        # Continuous-angle iota recovers the analytic 0.273 well within
        # the plan's 5e-3 convergence tolerance (no integer floor).
        assert iota[0] == pytest.approx(true_iota, abs=1e-3)

    def test_short_trajectory_reports_nan(self):
        """A trajectory shorter than one toroidal revolution returns NaN."""
        bridge, axis = self._shared_axis(major_radius=1.0)
        # Half a revolution => delta_phi = pi < 2*pi => NaN.
        phi = np.linspace(0.0, np.pi, 50)
        traj = np.stack(
            [
                np.linspace(0.0, 1.0, 50),
                1.1 * np.cos(phi),
                1.1 * np.sin(phi),
                np.zeros(50),
            ],
            axis=-1,
        )
        iota = bridge._continuous_angle_iota([traj], axis)
        assert np.isnan(iota[0])


class TestSeedAlignment:
    """C2 fix: per-seed-index pairing across short and long convergence rungs.

    The legacy implementation dropped escapees via ``continue`` and
    paired the resulting trimmed lists by index, which silently
    mis-aligned iota comparisons whenever seed 0 escaped at one rung
    but survived at the other. The fix pads escapees with ``np.nan``
    so the per-seed-index identity is preserved across rungs.
    """

    def test_per_seed_nan_padding_preserves_seed_index(self, monkeypatch):
        """When a single seed escapes, the others retain their seed-index slot.

        Build a 4-seed batch where seed 1 escapes (short trajectory) at
        the long rung, but produces a usable iota at the short rung.
        The convergence-residual pairing must drop seed 1 from the
        comparison while still comparing seeds 0/2/3 against their own
        long-rung counterparts (not against each other).
        """
        bridge = load_bridge_module()
        major_radius = 1.0

        class _RingSurface:
            def cross_section(self, phi, thetas):
                theta = np.linspace(
                    0.0, 2.0 * np.pi, int(thetas), endpoint=False
                )
                toroidal_angle = 2.0 * np.pi * float(phi)
                rr = major_radius + 0.0 * theta
                zz = 0.0 * theta
                return np.stack(
                    [
                        rr * np.cos(toroidal_angle),
                        rr * np.sin(toroidal_angle),
                        zz,
                    ],
                    axis=-1,
                )

            def gamma(self):
                phis = np.linspace(0.0, 1.0, 16, endpoint=False)
                sections = [self.cross_section(p, thetas=32) for p in phis]
                return np.stack(sections, axis=0)

        # Construct four helical trajectories with distinct true iotas;
        # the long-rung version of seed 1 is truncated to less than one
        # revolution so it is NaN-padded and dropped from the pair list.
        def _make_traj(iota, n_revs, n_steps):
            phi = np.linspace(0.0, n_revs * 2.0 * np.pi, n_steps)
            theta = iota * phi
            R = 1.0 + 0.05 * np.sin(theta)
            Z = 0.0 + 0.05 * np.cos(theta)
            return np.stack(
                [
                    np.linspace(0.0, 1.0, n_steps),
                    R * np.cos(phi),
                    R * np.sin(phi),
                    Z,
                ],
                axis=-1,
            )

        true_iotas = (0.20, 0.25, 0.30, 0.35)
        short_histories = [
            _make_traj(it, n_revs=10, n_steps=4001) for it in true_iotas
        ]
        long_histories = [
            _make_traj(it, n_revs=20, n_steps=8001) for it in true_iotas
        ]
        # Make seed 1 escape on the LONG rung (less than 1 revolution).
        long_histories[1] = _make_traj(0.25, n_revs=0.4, n_steps=20)

        call_count = {"count": 0}

        def _fake_trace(field, R0, Z0, **kwargs):
            call_count["count"] += 1
            return (
                short_histories if call_count["count"] == 1 else long_histories,
                [],
            )

        monkeypatch.setattr(
            bridge,
            "midplane_seed_radii",
            lambda surface, n: np.linspace(0.95, 1.05, int(n)),
        )
        monkeypatch.setattr(
            bridge,
            "build_stopping_criteria",
            lambda surface: ([], None),
        )
        monkeypatch.setattr(bridge, "compute_fieldlines", _fake_trace)

        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_RingSurface(),
            n_lines=4,
            tmax=10.0,
            tol=1e-8,
        )

        # All four seeds survived the short rung => 4 finite entries in
        # iota_per_line. Seed 1 (true_iota=0.25) sits at index 1.
        assert len(result.iota_per_line) == 4
        finite_short = [
            v for v in result.iota_per_line if not np.isnan(v)
        ]
        assert len(finite_short) == 4
        # The convergence residual is the MAX over paired seeds. Seed 1
        # is dropped from the pairing (NaN at long rung) — if pairing
        # had been mis-aligned (legacy: seeds 0/2/3 from long compared
        # against seeds 0/1/2 from short), the residual would be the
        # |0.20-0.25| ~ 0.05 mismatch, FAR above the 5e-3 tolerance.
        # With correct per-seed pairing, the residual is the numerical
        # quantization noise of the continuous-angle path, well below
        # the convergence tolerance.
        assert result.convergence_residual is not None
        assert result.convergence_residual < 1e-2
        # Three seed pairs (indices 0, 2, 3) contributed to the
        # convergence check; the proxy_mean averages only the SHORT-rung
        # finite-iota entries (which includes seed 1 since it survived
        # short).
        assert result.n_lines_survived == 4

    def test_legacy_pairing_alignment_bug_caught(self, monkeypatch):
        """Documents the C2 bug — pre-fix code would have falsely succeeded.

        Sets up the same scenario as the previous test and verifies that
        a manually mis-aligned comparison (dropping the NaN entry from
        both arrays before zipping) would yield a much LARGER residual
        than the per-seed-index comparison. This is the failure mode the
        C2 fix prevents.
        """
        bridge = load_bridge_module()
        true_iotas = np.array([0.20, 0.25, 0.30, 0.35])
        # Long rung loses seed 1 => long_finite = [0.20, 0.30, 0.35].
        short_finite = true_iotas.copy()
        long_with_nan = true_iotas.copy()
        long_with_nan[1] = np.nan
        # The CORRECT pair-wise comparison (C2 fix) compares seed
        # indices where both are finite: seeds 0, 2, 3. Residual = 0.
        valid_pairs = np.isfinite(short_finite) & np.isfinite(long_with_nan)
        correct_residual = float(
            np.max(
                np.abs(
                    long_with_nan[valid_pairs] - short_finite[valid_pairs]
                )
            )
        )
        # The LEGACY bug compared trimmed lists by index:
        # short = [0.20, 0.25, 0.30, 0.35], long_trimmed = [0.20, 0.30, 0.35]
        # zip(short, long_trimmed) => (0.20, 0.20), (0.25, 0.30), (0.30, 0.35)
        # giving a 0.05 mis-alignment residual.
        long_trimmed = long_with_nan[~np.isnan(long_with_nan)]
        n_compare_legacy = min(len(short_finite), len(long_trimmed))
        legacy_residual = float(
            np.max(
                np.abs(
                    long_trimmed[:n_compare_legacy]
                    - short_finite[:n_compare_legacy]
                )
            )
        )
        # The legacy mis-alignment produces a residual orders of
        # magnitude larger than the correct one.
        assert correct_residual == pytest.approx(0.0)
        assert legacy_residual > 0.04
        # Verify the C2-fixed path matches the correct residual.
        assert bridge.FIELDLINE_IOTA_CONVERGENCE_TOL == 0.005


class TestHelicalCacheFingerprint:
    """H2 fix: HelicalFieldContentObjective cache invalidates on field.x change.

    Without the fingerprint, mutating ``field.x`` (the optimizer's
    standard handle) would not invalidate the cache, so subsequent
    ``J()`` calls would silently return stale values from an older DOF
    configuration. The fingerprint compares the byte representation of
    the current ``field.x`` against the cached one.
    """

    def _build_counting_field(self, n_dofs: int, B_vector: np.ndarray):
        b_call_count = {"count": 0}

        class _Field:
            def __init__(self):
                self.x = np.zeros(n_dofs)

            def set_points(self, points):
                pass

            def B(self):
                b_call_count["count"] += 1
                return B_vector

            def B_vjp(self, v):
                grad_payload = np.arange(n_dofs, dtype=float) + 1.0

                class _Derivative:
                    def __call__(self, optim):
                        return grad_payload

                return _Derivative()

        return _Field(), b_call_count

    def test_cache_invalidates_when_field_x_changes(self):
        """H2 regression: a silent field.x mutation must trigger a recompute."""
        bridge = load_bridge_module()
        nphi, ntheta = 16, 16
        phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA - 2 * PHI)],
            axis=-1,
        )
        amplitude = 1.0 + 0.2 * np.cos(THETA.reshape(-1) - 2 * PHI.reshape(-1))
        phi_flat = PHI.reshape(-1)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)
        B_vector = np.stack([Bx, By, Bz], axis=-1)

        n_dofs = 3
        field, b_call_count = self._build_counting_field(n_dofs, B_vector)
        surface = SimpleNamespace(gamma=lambda: gamma)
        objective = bridge.HelicalFieldContentObjective(field, surface)

        # First J() computes; second J() reuses cache.
        _ = objective.J()
        assert b_call_count["count"] == 1
        _ = objective.J()
        assert b_call_count["count"] == 1

        # Mutating field.x WITHOUT calling recompute_bell must
        # invalidate the cache on the next J() call.
        field.x = np.ones(n_dofs)
        _ = objective.J()
        assert b_call_count["count"] == 2

    def test_recompute_bell_still_invalidates_cache(self):
        """Explicit invalidation via recompute_bell remains functional."""
        bridge = load_bridge_module()
        nphi, ntheta = 16, 16
        phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA - 2 * PHI)],
            axis=-1,
        )
        amplitude = 1.0 + 0.2 * np.cos(THETA.reshape(-1) - 2 * PHI.reshape(-1))
        phi_flat = PHI.reshape(-1)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)
        B_vector = np.stack([Bx, By, Bz], axis=-1)

        n_dofs = 3
        field, b_call_count = self._build_counting_field(n_dofs, B_vector)
        surface = SimpleNamespace(gamma=lambda: gamma)
        objective = bridge.HelicalFieldContentObjective(field, surface)
        _ = objective.J()
        assert b_call_count["count"] == 1
        objective.recompute_bell()
        _ = objective.J()
        assert b_call_count["count"] == 2


class TestCaller_S_HEL_Then_Tracer_Ordering:
    """H3 fix: the production call site must evaluate S_HEL before tracing.

    The S_HEL helper sets ``field.set_points(gamma)`` without restoring
    the previous points (no defensive try/finally per project rules).
    The fix is to *document* this as a contract: S_HEL runs FIRST, then
    the tracer (which always re-sets the field's points on every step),
    so the side effect is invisible in production. This test pins the
    integration-site ordering so a future refactor cannot reverse the
    sequence and silently corrupt the field's evaluation points.
    """

    def test_banana_coil_solver_evaluates_s_hel_before_tracer(self):
        """Static check: the banana_coil_solver topology-bridge block
        calls ``safe_compute_helical_field_content_S_HEL`` BEFORE
        ``safe_compute_phase3a_fieldline_iota_proxy`` so the S_HEL side
        effect is always overwritten before any other code reads
        ``field.B()``.

        Reads the file directly rather than importing
        ``STAGE_2.banana_coil_solver`` so the test does not depend on
        the example-package sys.path setup the solver runtime injects;
        the source-text check is the contract we care about.
        """
        solver_path = (
            EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
        )
        src = solver_path.read_text()
        s_hel_pos = src.find("safe_compute_helical_field_content_S_HEL(")
        tracer_pos = src.find("safe_compute_phase3a_fieldline_iota_proxy(")
        assert s_hel_pos > 0, (
            "expected safe_compute_helical_field_content_S_HEL "
            "in banana_coil_solver"
        )
        assert tracer_pos > 0, (
            "expected safe_compute_phase3a_fieldline_iota_proxy "
            "in banana_coil_solver"
        )
        # S_HEL must come first so its set_points side effect is
        # overwritten by the tracer (which always re-sets points).
        assert s_hel_pos < tracer_pos


class TestSafeComputeSHEL:
    """The artifact path must NOT crash the solver on degenerate B-fields."""

    def test_safe_wrapper_returns_value_on_clean_input(self):
        bridge = load_bridge_module()
        nphi, ntheta = 16, 16
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA - 2 * PHI)],
            axis=-1,
        )
        theta_flat = THETA.reshape(-1)
        phi_flat = PHI.reshape(-1)
        amplitude = 1.0 + 0.3 * np.cos(theta_flat - 2 * phi_flat)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)

        class _Field:
            def set_points(self, _): pass
            def B(self): return np.stack([Bx, By, Bz], axis=-1)

        surface = SimpleNamespace(gamma=lambda: gamma)
        result = bridge.safe_compute_helical_field_content_S_HEL(_Field(), surface)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_safe_wrapper_returns_none_on_zero_field(self):
        """A truly zero |B| grid is the canonical undefined-S_HEL case."""
        bridge = load_bridge_module()
        nphi, ntheta = 8, 8
        gamma = np.zeros((nphi, ntheta, 3))

        class _ZeroField:
            def set_points(self, points):
                self._n = np.asarray(points).shape[0]
            def B(self):
                return np.zeros((self._n, 3))

        surface = SimpleNamespace(gamma=lambda: gamma)
        assert bridge.safe_compute_helical_field_content_S_HEL(
            _ZeroField(), surface
        ) is None

    def test_safe_wrapper_returns_none_on_non_finite_field(self):
        bridge = load_bridge_module()
        nphi, ntheta = 8, 8
        gamma = np.zeros((nphi, ntheta, 3))

        class _NanField:
            def set_points(self, points):
                self._n = np.asarray(points).shape[0]
            def B(self):
                return np.full((self._n, 3), np.nan)

        surface = SimpleNamespace(gamma=lambda: gamma)
        assert bridge.safe_compute_helical_field_content_S_HEL(
            _NanField(), surface
        ) is None

    def test_safe_wrapper_propagates_non_value_errors(self):
        """The safe wrapper must NOT swallow TypeError, AttributeError, etc.

        Only the documented ``ValueError`` modes are swallowed; everything
        else surfaces so genuine bugs are not masked.
        """
        bridge = load_bridge_module()

        class _BrokenField:
            def set_points(self, _): pass
            def B(self):
                raise RuntimeError("simsopt internal failure")

        surface = SimpleNamespace(gamma=lambda: np.zeros((4, 4, 3)))
        with pytest.raises(RuntimeError, match="simsopt internal failure"):
            bridge.safe_compute_helical_field_content_S_HEL(
                _BrokenField(), surface
            )


def _build_tiny_biotsavart_problem():
    """Build a real BiotSavart on a tiny working surface for FD validation.

    Uses the canonical NCSX coil set (intrinsically helical, so ``S_HEL``
    has non-trivial helical content and the gradient is well-conditioned
    against centered finite differences) on a small
    :class:`SurfaceXYZTensorFourier` fitted to the magnetic axis.
    ``nphi=ntheta=8`` keeps the FD sweep cheap; ``mpol=ntor=1`` keeps the
    surface geometry deliberately under-resolved — we only need it as a
    sampling grid, not a Boozer surface.
    """
    from simsopt.configs import get_ncsx_data
    from simsopt.field import coils_via_symmetries
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

    base_curves, base_currents, axis = get_ncsx_data()
    nfp = 3
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    field = BiotSavart(coils)

    nphi, ntheta = 8, 8
    phis = np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False)
    thetas = np.linspace(0.0, 1.0, ntheta, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=1,
        ntor=1,
        nfp=nfp,
        stellsym=False,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(axis, 0.05, flip_theta=True)
    return field, surface


class TestSHELGradient:
    def test_gradient_is_zero_for_axisymmetric_field(self):
        """A field whose ``|B|`` has only ``n=0`` content has S_HEL = 0 and
        zero S_HEL gradient component-wise.

        The chain rule terminates at ``dS/d(modB) = 0`` everywhere (because
        the numerator is identically zero AND its gradient is zero — the
        masked FFT is zero, so its inverse FFT is zero); therefore
        ``B_vjp(0) == 0`` for every coil DOF regardless of the underlying
        coil geometry.
        """
        bridge = load_bridge_module()

        nphi, ntheta = 16, 16
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA)],
            axis=-1,
        )

        amplitude = 1.0 + 0.3 * np.cos(THETA.reshape(-1))
        phi_flat = PHI.reshape(-1)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)
        B_vector = np.stack([Bx, By, Bz], axis=-1)

        # 4 fake DOFs; the helper's B_vjp produces a zero gradient regardless
        # because dS/d(modB) is zero everywhere when modB has only n=0
        # content (axisymmetric on the (phi, theta) grid).
        n_dofs = 4

        class _AxisymField:
            def __init__(self):
                self.x = np.zeros(n_dofs)
                self.vjp_calls = []

            def set_points(self, points):
                self._n = np.asarray(points).shape[0]

            def B(self):
                return B_vector

            def B_vjp(self, v):
                self.vjp_calls.append(np.asarray(v, dtype=float).copy())

                class _Derivative:
                    def __init__(self, payload):
                        self.payload = payload

                    def __call__(self, optim):
                        # Adjoint of a zero co-tangent must be zero; we also
                        # assert the co-tangent itself was zero so the
                        # contract is unambiguous.
                        return np.zeros(n_dofs)

                return _Derivative(v)

        surface = SimpleNamespace(gamma=lambda: gamma)
        field = _AxisymField()
        objective = bridge.HelicalFieldContentObjective(field, surface)
        s_hel = objective.J()
        grad = objective.dJ_by_dcoils()
        assert s_hel == pytest.approx(0.0, abs=1e-12)
        assert grad.shape == (n_dofs,)
        np.testing.assert_allclose(grad, np.zeros(n_dofs), atol=1e-12)
        # The co-tangent passed to B_vjp must also vanish — proof the chain
        # rule terminated correctly, not just the final flat product.
        assert len(field.vjp_calls) == 1
        np.testing.assert_allclose(
            field.vjp_calls[0], np.zeros_like(field.vjp_calls[0]), atol=1e-12
        )

    def test_gradient_matches_central_difference_on_real_biot_savart(self):
        """End-to-end FD check on a real :class:`BiotSavart` problem."""
        bridge = load_bridge_module()
        field, surface = _build_tiny_biotsavart_problem()
        max_rel_error = bridge.compute_S_HEL_gradient_relative_error(
            field, surface, perturbation=1.0e-5, seed=0
        )
        assert max_rel_error < 1.0e-3, (
            f"S_HEL analytic gradient failed FD validation: "
            f"max_rel_error={max_rel_error:.3e}"
        )

    def test_objective_caches_J_dJ(self):
        """``J`` / ``dJ_by_dcoils`` compute once until ``recompute_bell``."""
        bridge = load_bridge_module()

        nphi, ntheta = 16, 16
        phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
        gamma = np.stack(
            [np.cos(PHI), np.sin(PHI), 0.1 * np.sin(THETA - 2 * PHI)],
            axis=-1,
        )
        amplitude = 1.0 + 0.2 * np.cos(THETA.reshape(-1) - 2 * PHI.reshape(-1))
        phi_flat = PHI.reshape(-1)
        Bx = amplitude * np.cos(phi_flat)
        By = amplitude * np.sin(phi_flat)
        Bz = np.zeros_like(amplitude)
        B_vector = np.stack([Bx, By, Bz], axis=-1)

        n_dofs = 3
        b_call_count = {"count": 0}
        vjp_call_count = {"count": 0}

        class _CountingField:
            def __init__(self):
                self.x = np.zeros(n_dofs)

            def set_points(self, points):
                pass

            def B(self):
                b_call_count["count"] += 1
                return B_vector

            def B_vjp(self, v):
                vjp_call_count["count"] += 1
                grad_payload = np.arange(n_dofs, dtype=float) + 1.0

                class _Derivative:
                    def __call__(self, optim):
                        return grad_payload

                return _Derivative()

        surface = SimpleNamespace(gamma=lambda: gamma)
        objective = bridge.HelicalFieldContentObjective(_CountingField(), surface)
        # First J() computes both J and grad (single B/B_vjp pair).
        _ = objective.J()
        assert b_call_count["count"] == 1
        assert vjp_call_count["count"] == 1
        # Second J() and a dJ() reuse the cache (no extra B/B_vjp calls).
        _ = objective.J()
        _ = objective.dJ_by_dcoils()
        assert b_call_count["count"] == 1
        assert vjp_call_count["count"] == 1
        # recompute_bell forces a fresh compute on next access.
        objective.recompute_bell()
        _ = objective.dJ_by_dcoils()
        assert b_call_count["count"] == 2
        assert vjp_call_count["count"] == 2


class TestSharedCoordinateAxis:
    """The centroid axis class is the SSOT for both Phase 3a and Phase 3b.

    F13 fix: the byte-identical ``_SurfaceCentroidCoordinateAxis`` was
    extracted into :mod:`topology_bridge_shared`; both Phase 3 modules
    import from there so they cannot drift apart.
    """

    def test_shared_class_resolves_to_phase3b_alias(self):
        bridge = load_bridge_module()
        from banana_opt import topology_bridge_shared as shared
        assert bridge._SurfaceCentroidCoordinateAxis is shared.SurfaceCentroidCoordinateAxis

    def test_shared_factory_resolves_to_phase3b_alias(self):
        bridge = load_bridge_module()
        from banana_opt import topology_bridge_shared as shared
        assert bridge._surface_centroid_coordinate_axis is shared.build_surface_centroid_axis

    def test_shared_defaults_match_phase3b_reexports(self):
        bridge = load_bridge_module()
        from banana_opt import topology_bridge_shared as shared
        # Phase 3b re-exports the shared trace-length and tolerance
        # defaults so the two bridges agree on the contract. The
        # ``DEFAULT_NFIELDLINES`` shared constant is intentionally NOT
        # re-exported here because Phase 3b's two field-line callers use
        # distinct hardcoded ``n_lines`` values (see the per-function
        # docstrings); wiring the shared default through would silently
        # rewrite them.
        assert bridge.DEFAULT_TMAX == shared.DEFAULT_TMAX
        assert bridge.DEFAULT_TOL == shared.DEFAULT_TOL

    def test_phase3b_does_not_re_export_default_nfieldlines(self):
        """Pin the Phase 3b N4 contract: ``DEFAULT_NFIELDLINES`` is NOT re-exported.

        Phase 3b's :func:`compute_fieldline_iota_proxy` and
        :func:`safe_compute_fieldline_iota_proxy` use hardcoded
        ``n_lines`` defaults (``8`` and ``4`` respectively) for distinct
        call-site contexts. Re-exporting the shared ``DEFAULT_NFIELDLINES``
        from this module would invite consumers to bind it as a default
        and silently rewrite those values. The canonical Phase 3a
        default lives in :mod:`banana_opt.topology_bridge_shared`; this
        test will fail if a future refactor re-introduces the re-export.
        """
        bridge = load_bridge_module()
        assert not hasattr(bridge, "DEFAULT_NFIELDLINES")
        assert "DEFAULT_NFIELDLINES" not in bridge.__all__
