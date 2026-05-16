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
    def test_none_results_emit_all_keys(self):
        bridge = load_bridge_module()
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=None,
            fieldline_result=None,
            s_hel_objective_weight=None,
            iota_target=0.2,
        )
        assert set(fields) == {
            "FIELDLINE_IOTA_PROXY",
            "FIELDLINE_IOTA_PROXY_VALID",
            "HELICAL_FIELD_CONTENT",
            "S_HEL_OBJECTIVE_WEIGHT",
            "PRE_BOOZER_TOPOLOGY_SCORE",
        }
        assert all(value is None for value in fields.values())

    def test_concrete_results_populate_keys(self):
        bridge = load_bridge_module()
        fieldline = bridge.FieldlineIotaProxyResult(
            iota_proxy_mean=0.195,
            iota_proxy_std=0.003,
            iota_per_line=(0.197, 0.193, 0.195),
            n_lines_seeded=8,
            n_lines_survived=3,
            tmax=200.0,
            valid=True,
            convergence_residual=0.0012,
            invalid_reason=None,
        )
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.42,
            fieldline_result=fieldline,
            s_hel_objective_weight=1.0e-3,
            iota_target=0.2,
        )
        assert fields["FIELDLINE_IOTA_PROXY"] == pytest.approx(0.195)
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is True
        assert fields["HELICAL_FIELD_CONTENT"] == pytest.approx(0.42)
        assert fields["S_HEL_OBJECTIVE_WEIGHT"] == pytest.approx(1.0e-3)
        # iota_proxy is within 0.005 of target so proximity ≈ 0.995.
        # PRE_BOOZER_TOPOLOGY_SCORE = 0.42 * 0.995 ≈ 0.4179.
        assert fields["PRE_BOOZER_TOPOLOGY_SCORE"] == pytest.approx(
            0.42 * (1.0 - abs(0.195 - 0.2)), abs=1e-9
        )

    def test_invalid_fieldline_proxy_is_treated_as_unavailable(self):
        bridge = load_bridge_module()
        fieldline = bridge.FieldlineIotaProxyResult(
            iota_proxy_mean=0.1,
            iota_proxy_std=0.0,
            iota_per_line=(0.1,),
            n_lines_seeded=8,
            n_lines_survived=1,
            tmax=200.0,
            valid=False,
            convergence_residual=0.5,
            invalid_reason="convergence_tol_exceeded",
        )
        fields = bridge.boozer_topology_bridge_artifact_fields(
            s_hel=0.5,
            fieldline_result=fieldline,
            s_hel_objective_weight=None,
            iota_target=0.2,
        )
        # S_HEL alone (invalid proxy is treated as unavailable).
        assert fields["PRE_BOOZER_TOPOLOGY_SCORE"] == pytest.approx(0.5)
        # But the raw proxy value is still surfaced for inspection.
        assert fields["FIELDLINE_IOTA_PROXY"] == pytest.approx(0.1)
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is False


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
