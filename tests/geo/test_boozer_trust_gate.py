"""Unit tests for the Phase 1 Boozer residual trust gate.

The trust gate is the SSOT that decides whether the Stage 2 iota objective
can use a Boozer-derived gradient. These tests pin the reconstruction of the
constrained-residual norm, the failure-mode taxonomy, and the way the
optimizer entry point honors ``iota_objective_active``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from alm_utils import _updated_nonnegative_multipliers  # noqa: E402


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def load_stage2_objectives_module():
    return importlib.import_module("banana_opt.stage2_objectives")


def _fake_label(value: float, target: float) -> SimpleNamespace:
    return SimpleNamespace(J=lambda: float(value), targetlabel=float(target))


def _fake_surface(*, stellsym: bool = True, gamma_z0: float = 0.0):
    def gamma():
        return np.full((1, 1, 3), 0.0, dtype=float).reshape((1, 1, 3)) + np.array(
            [0.0, 0.0, gamma_z0]
        )

    return SimpleNamespace(stellsym=stellsym, gamma=gamma)


def _make_ls_boozer_surface(
    *,
    residual: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    newton_tol: float = 1e-11,
    iota: float = 0.2,
):
    """Synthesize a BoozerSurface stand-in whose ``res`` matches the LS path.

    The LS-path trust gate reads ``res['jacobian']`` (the gradient norm the
    upstream Newton success check uses). When ``jacobian`` is omitted,
    ``residual`` is used as a stand-in so older test cases that exercised the
    residual-norm semantics keep their original numerics meaningful — the
    trust gate just measures whatever vector lives at the gradient slot.
    """

    if residual is None and jacobian is None:
        res = None
    else:
        gradient_source = jacobian if jacobian is not None else residual
        res = {
            "iota": iota,
            "G": None,
            "type": "ls",
            "success": True,
            "jacobian": np.asarray(gradient_source, dtype=float).copy(),
        }
        if residual is not None:
            res["residual"] = np.asarray(residual, dtype=float).copy()
    return SimpleNamespace(
        res=res,
        boozer_type="ls",
        options={"newton_tol": newton_tol},
        surface=_fake_surface(),
        label=_fake_label(0.0, 0.0),
        targetlabel=0.0,
    )


def _make_exact_boozer_surface(
    *,
    raw_residual: np.ndarray,
    mask: np.ndarray,
    label_value: float = 0.0,
    target_label: float = 0.0,
    iota: float = 0.2,
    newton_tol: float = 1e-13,
    stellsym: bool = True,
    gamma_z0: float = 0.0,
):
    res = {
        "residual": np.asarray(raw_residual, dtype=float).copy(),
        "mask": np.asarray(mask, dtype=bool).copy(),
        "iota": iota,
        "G": 1.0,
        "type": "exact",
        "success": True,
    }
    surface = _fake_surface(stellsym=stellsym, gamma_z0=gamma_z0)
    return SimpleNamespace(
        res=res,
        boozer_type="exact",
        options={"newton_tol": newton_tol},
        surface=surface,
        label=_fake_label(label_value, target_label),
        targetlabel=float(target_label),
    )


class TestConstrainedResidualReconstruction:
    def test_returns_none_when_no_solve_yet(self):
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(res=None, options={}, surface=None)
        assert handoff.compute_boozer_constrained_residual_norm(boozer_surface) is None

    def test_ls_path_returns_norm_of_gradient_not_residual(self):
        """LS Newton success uses ``||jacobian||``, not ``||residual||``.

        Regression for the Crucible-flagged trust-gate inversion: at LS
        convergence the residual sits at a non-zero minimum while the
        gradient vanishes, so the residual-norm path would mis-flag every
        converged solve as untrusted.
        """
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface(
            jacobian=np.array([3.0, 4.0]),  # ||grad|| = 5
            residual=np.array([100.0, 200.0]),  # ||r|| ~ 224, ignored
        )
        norm = handoff.compute_boozer_constrained_residual_norm(boozer_surface)
        assert norm == pytest.approx(5.0)

    def test_ls_path_zero_gradient_means_converged(self):
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface(jacobian=np.zeros(7))
        assert handoff.compute_boozer_constrained_residual_norm(boozer_surface) == 0.0

    def test_ls_path_returns_none_when_no_gradient_keys(self):
        """LS-path: neither 'jacobian' nor 'gradient' present => None.

        H8 fix: the LS-family branch resolves the gradient vector by
        explicit key presence (no fallback). Only a residual entry but
        no gradient entry is the "diagnostic-unavailable" case the
        trust gate must report as ``None``.
        """
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(
            res={
                "iota": 0.2,
                "G": None,
                "type": "ls",
                "success": True,
                "residual": np.array([1.0, 2.0]),
            },
            boozer_type="ls",
            options={"newton_tol": 1e-11},
            surface=_fake_surface(),
            label=_fake_label(0.0, 0.0),
            targetlabel=0.0,
        )
        assert handoff.compute_boozer_constrained_residual_norm(boozer_surface) is None

    def test_ls_path_lbfgs_gradient_key_is_read(self):
        """LBFGS LS path persists the gradient under ``res['gradient']``.

        H8 fix: the LBFGS solver (minimize_boozer_penalty_constraints_LBFGS,
        boozersurface.py:565-570) writes the optimizer gradient as
        ``res['gradient'] = res.jac`` (not ``res['jacobian']``). Both
        LS-family writers are gradient vectors of the scalarized LS
        objective; the trust gate must accept either key.
        """
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(
            res={
                "iota": 0.2,
                "G": None,
                "type": "ls",
                "success": True,
                "gradient": np.array([6.0, 8.0]),  # ||grad|| = 10
            },
            boozer_type="ls",
            options={"newton_tol": 1e-11},
            surface=_fake_surface(),
            label=_fake_label(0.0, 0.0),
            targetlabel=0.0,
        )
        norm = handoff.compute_boozer_constrained_residual_norm(boozer_surface)
        assert norm == pytest.approx(10.0)

    def test_ls_path_jacobian_key_takes_precedence_over_gradient_key(self):
        """When both keys are present, ``jacobian`` (LS Newton) wins.

        LS Newton and LBFGS are mutually exclusive writers in the
        upstream solver — only one populates ``res`` at a time. The
        ``jacobian`` key takes precedence in the explicit key-presence
        check so a future writer that surfaces both keys still produces
        a deterministic result. (Both are gradient vectors of the same
        scalarized LS objective, so the choice is semantic, not
        numerical.)
        """
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(
            res={
                "iota": 0.2,
                "G": None,
                "type": "ls",
                "success": True,
                "jacobian": np.array([3.0, 4.0]),  # ||grad|| = 5
                "gradient": np.array([6.0, 8.0]),  # ||grad|| = 10, ignored
            },
            boozer_type="ls",
            options={"newton_tol": 1e-11},
            surface=_fake_surface(),
            label=_fake_label(0.0, 0.0),
            targetlabel=0.0,
        )
        norm = handoff.compute_boozer_constrained_residual_norm(boozer_surface)
        assert norm == pytest.approx(5.0)

    def test_exact_path_reconstructs_with_mask_and_label(self):
        handoff = load_handoff_module()
        # 4-component residual, mask keeps indices 0 and 2.
        raw = np.array([1.0, 99.0, 2.0, 99.0])
        mask = np.array([True, False, True, False])
        # Label residual = label.J() - targetlabel = 5.0 - 3.0 = 2.0
        boozer_surface = _make_exact_boozer_surface(
            raw_residual=raw,
            mask=mask,
            label_value=5.0,
            target_label=3.0,
            stellsym=True,
        )
        # Constrained vector = [1.0, 2.0, 2.0]; norm = sqrt(1 + 4 + 4) = 3
        norm = handoff.compute_boozer_constrained_residual_norm(boozer_surface)
        assert norm == pytest.approx(3.0)

    def test_exact_path_non_stellsym_appends_z0(self):
        handoff = load_handoff_module()
        raw = np.array([3.0, 99.0])
        mask = np.array([True, False])
        # Label = 0, z0 = 4. Norm = sqrt(9 + 0 + 16) = 5.
        boozer_surface = _make_exact_boozer_surface(
            raw_residual=raw,
            mask=mask,
            label_value=0.0,
            target_label=0.0,
            stellsym=False,
            gamma_z0=4.0,
        )
        norm = handoff.compute_boozer_constrained_residual_norm(boozer_surface)
        assert norm == pytest.approx(5.0)

    def test_exact_path_missing_mask_returns_none(self):
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(
            res={
                "residual": np.array([1.0, 2.0]),
                "iota": 0.2,
                "G": 1.0,
                "type": "exact",
                "success": True,
            },
            boozer_type="exact",
            options={"newton_tol": 1e-13},
            surface=_fake_surface(),
            label=_fake_label(0.0, 0.0),
            targetlabel=0.0,
        )
        assert handoff.compute_boozer_constrained_residual_norm(boozer_surface) is None


class TestBoozerTrustState:
    def test_solve_failed_is_untrusted_and_inactive(self):
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface()
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=False, self_intersecting=None
        )
        assert state.boozer_trusted is False
        assert state.iota_objective_active is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_SOLVE_FAILED
        assert state.constrained_residual_norm is None

    def test_self_intersecting_is_untrusted(self):
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface(jacobian=np.zeros(3))
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=True
        )
        assert state.boozer_trusted is False
        assert state.iota_objective_active is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_SELF_INTERSECTING

    def test_gradient_within_tolerance_is_trusted(self):
        handoff = load_handoff_module()
        # Newton tol 1e-11, trust tol = 1e-10. ||grad|| = 5e-11 is converged.
        boozer_surface = _make_ls_boozer_surface(
            jacobian=np.array([3e-11, 4e-11]),  # ||grad|| = 5e-11
        )
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        assert state.boozer_trusted is True
        assert state.iota_objective_active is True
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_OK
        assert state.constrained_residual_norm == pytest.approx(5e-11)
        assert state.trust_tol == pytest.approx(1e-10)

    def test_gradient_above_tolerance_is_untrusted(self):
        handoff = load_handoff_module()
        # Newton tol 1e-11, trust tol = 1e-10. ||grad|| = 1e-3 is far out.
        boozer_surface = _make_ls_boozer_surface(
            jacobian=np.array([1e-3, 0.0]),
        )
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        assert state.boozer_trusted is False
        assert state.iota_objective_active is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_RESIDUAL_TOO_LARGE
        assert state.constrained_residual_norm == pytest.approx(1e-3)

    def test_unavailable_gradient_is_untrusted(self):
        handoff = load_handoff_module()
        boozer_surface = SimpleNamespace(
            res={"iota": 0.2, "G": 1.0, "type": "ls", "success": True},
            boozer_type="ls",
            options={"newton_tol": 1e-11},
            surface=_fake_surface(),
            label=_fake_label(0.0, 0.0),
            targetlabel=0.0,
        )
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        assert state.boozer_trusted is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_RESIDUAL_UNAVAILABLE

    def test_nonphysical_iota_disables_trust(self):
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface(
            jacobian=np.array([1e-12]),
            iota=42.0,  # well outside the HBT campaign band
        )
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        assert state.boozer_trusted is False
        assert state.iota_objective_active is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_IOTA_NONPHYSICAL


class TestArtifactFields:
    def test_none_trust_state_emits_all_keys_as_none(self):
        handoff = load_handoff_module()
        fields = handoff.boozer_trust_artifact_fields(None)
        assert set(fields) == {
            "BOOZER_SOLVE_SUCCESS",
            "BOOZER_SELF_INTERSECTING",
            "BOOZER_CONSTRAINED_RESIDUAL_NORM",
            "BOOZER_TRUSTED",
            "IOTA_OBJECTIVE_ACTIVE",
            "BOOZER_TRUST_REASON",
            "BOOZER_TRUST_TOL",
        }
        assert all(value is None for value in fields.values())

    def test_trusted_state_emits_concrete_values(self):
        handoff = load_handoff_module()
        boozer_surface = _make_ls_boozer_surface(jacobian=np.array([1e-13]))
        state = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        fields = handoff.boozer_trust_artifact_fields(state)
        assert fields["BOOZER_SOLVE_SUCCESS"] is True
        assert fields["BOOZER_SELF_INTERSECTING"] is False
        assert fields["BOOZER_TRUSTED"] is True
        assert fields["IOTA_OBJECTIVE_ACTIVE"] is True
        assert fields["BOOZER_TRUST_REASON"] == handoff.BOOZER_TRUST_REASON_OK
        assert fields["BOOZER_TRUST_TOL"] == pytest.approx(1e-10)
        assert fields["BOOZER_CONSTRAINED_RESIDUAL_NORM"] == pytest.approx(1e-13)


class TestStage2IotaStateTrustPropagation:
    def test_default_state_is_trusted(self):
        objectives = load_stage2_objectives_module()
        state = objectives.Stage2IotaState(
            iota=0.2, penalty=0.0, abs_error=0.0, feasible=True
        )
        assert state.boozer_trusted is True
        assert state.iota_objective_active is True

    def test_build_state_threads_untrusted_residual(self):
        handoff = load_handoff_module()
        objectives = load_stage2_objectives_module()
        boozer_surface = _make_ls_boozer_surface(jacobian=np.array([1.0]))
        trust = handoff.compute_boozer_trust_state(
            boozer_surface, solve_success=True, self_intersecting=False
        )
        # ``_build_stage2_iota_state_from_iota`` is the path used when the
        # guarded evaluator has a failed/untrusted result.
        state = objectives._build_stage2_iota_state_from_iota(
            0.2,
            target=0.2,
            tolerance=0.01,
            solve_failed=False,
            trust_state=trust,
        )
        assert state.boozer_trusted is False
        assert state.iota_objective_active is False
        assert state.trust_reason == handoff.BOOZER_TRUST_REASON_RESIDUAL_TOO_LARGE


class TestStage2FunHonorsTrust:
    def test_solve_failed_routes_to_reject_sentinel(self):
        objectives = load_stage2_objectives_module()

        base_J = 0.4
        base_grad = np.array([1.0, 2.0])

        class _FixedJF:
            def __init__(self):
                self.x = np.array([0.0, 0.0])

            def J(self):
                return base_J

            def dJ(self):
                return base_grad.copy()

        runtime = SimpleNamespace(
            mode="soft",
            weight=1.0,
            penalty_threshold=0.5,
            effective_weight=None,
            last_state=None,
            stats=SimpleNamespace(),
            iota_term=None,
            penalty_objective=None,
            target=0.2,
            tolerance=0.01,
            guarded_boozer_evaluator=None,
        )

        # Inject a failed evaluation. ``solve_failed=True`` mimics a true
        # Boozer-domain failure: the reject sentinel adds +1 to base flux
        # (because base_J < 1) and leaves the gradient unchanged.
        runtime.last_state = objectives.Stage2IotaState(
            iota=0.0,
            penalty=0.0,
            abs_error=0.2,
            feasible=False,
            solve_failed=True,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason="solve_failed",
        )

        # Monkey-patch evaluate_stage2_iota to return our pre-baked state.
        original_evaluate = objectives.evaluate_stage2_iota

        def _fake_evaluate(rt):
            return objectives.Stage2IotaEvaluation(
                state=runtime.last_state,
                penalty_grad=np.zeros_like(base_grad),
            )

        objectives.evaluate_stage2_iota = _fake_evaluate
        try:
            fake_surface = SimpleNamespace(
                unitnormal=lambda: np.zeros((1, 1, 3)),
                gamma=lambda: np.zeros((1, 1, 3)),
            )
            fake_bs = SimpleNamespace(
                set_points=lambda points: None,
                clear_cached_properties=lambda: None,
                B=lambda: np.zeros((1, 1, 3)),
            )
            fun = objectives.make_stage2_fun(
                JF=_FixedJF(),
                new_bs=fake_bs,
                new_surf=fake_surface,
                Jf=SimpleNamespace(J=lambda: 0.0),
                Jls=SimpleNamespace(J=lambda: 0.0),
                Jccdist=SimpleNamespace(shortest_distance=lambda: 0.0),
                Jc=SimpleNamespace(J=lambda: 0.0),
                stage2_iota_runtime=runtime,
            )
            J, grad = fun(np.array([0.0, 0.0]))
        finally:
            objectives.evaluate_stage2_iota = original_evaluate
        # Reject sentinel: when base_J < 1 it returns base_J + 1, grad unchanged
        assert J == pytest.approx(base_J + 1.0)
        np.testing.assert_allclose(grad, base_grad)

    def test_untrusted_but_evaluable_leaves_base_objective_unchanged(self):
        objectives = load_stage2_objectives_module()

        base_J = 0.4
        base_grad = np.array([1.0, 2.0])

        class _FixedJF:
            def __init__(self):
                self.x = np.array([0.0, 0.0])

            def J(self):
                return base_J

            def dJ(self):
                return base_grad.copy()

        runtime = SimpleNamespace(
            mode="soft",
            weight=1.0,
            penalty_threshold=0.5,
            effective_weight=None,
            last_state=None,
            stats=SimpleNamespace(),
            iota_term=None,
            penalty_objective=None,
            target=0.2,
            tolerance=0.01,
            guarded_boozer_evaluator=None,
        )

        runtime.last_state = objectives.Stage2IotaState(
            iota=0.2,
            penalty=0.01,
            abs_error=0.05,
            feasible=False,
            solve_failed=False,
            boozer_trusted=False,
            iota_objective_active=False,
            trust_reason="residual_too_large",
        )
        # Penalty gradient that *would* push the optimizer if it were applied.
        penalty_grad = np.array([100.0, 200.0])

        original_evaluate = objectives.evaluate_stage2_iota

        def _fake_evaluate(rt):
            return objectives.Stage2IotaEvaluation(
                state=runtime.last_state,
                penalty_grad=penalty_grad,
            )

        objectives.evaluate_stage2_iota = _fake_evaluate
        try:
            fake_surface = SimpleNamespace(
                unitnormal=lambda: np.zeros((1, 1, 3)),
                gamma=lambda: np.zeros((1, 1, 3)),
            )
            fake_bs = SimpleNamespace(
                set_points=lambda points: None,
                clear_cached_properties=lambda: None,
                B=lambda: np.zeros((1, 1, 3)),
            )
            fun = objectives.make_stage2_fun(
                JF=_FixedJF(),
                new_bs=fake_bs,
                new_surf=fake_surface,
                Jf=SimpleNamespace(J=lambda: 0.0),
                Jls=SimpleNamespace(J=lambda: 0.0),
                Jccdist=SimpleNamespace(shortest_distance=lambda: 0.0),
                Jc=SimpleNamespace(J=lambda: 0.0),
                stage2_iota_runtime=runtime,
            )
            J, grad = fun(np.array([0.0, 0.0]))
        finally:
            objectives.evaluate_stage2_iota = original_evaluate
        # Untrusted-but-evaluable: base J and grad untouched (no doubling,
        # no iota penalty injection).
        assert J == pytest.approx(base_J)
        np.testing.assert_allclose(grad, base_grad)


# ---------------------------------------------------------------------------
# ALM-lane helpers (mirroring the soft-lane synthetic Stage 2 evaluator state
# above). The ALM path drives the same iota_objective_active=False trust
# decision through ``evaluate_stage2_alm_problem`` instead of
# ``make_stage2_fun``; the test fakes here are the minimal synthetic Stage 2
# evaluator state (no real BoozerSurface) needed to exercise the lane.


class _AlmFakeBaseObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)
        self.x = None

    def J(self):
        return self._value

    def dJ(self):
        return self._grad.copy()


class _AlmFakeScalarObjective:
    def __init__(self, value):
        self._value = float(value)

    def J(self):
        return self._value


class _AlmFakeLengthObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda _objective: self._grad.copy()


class _AlmFakeWidthObjective(_AlmFakeLengthObjective):
    pass


class _AlmFakeSelfIntersectObjective:
    def __init__(self, value, grad, shortest_self_distance=0.5):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)
        self._shortest_self_distance = float(shortest_self_distance)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda _objective: self._grad.copy()

    def shortest_self_distance(self):
        return self._shortest_self_distance


class _AlmFakeCurveDistance:
    def __init__(self, minimum_distance, shortest_distance):
        self.minimum_distance = float(minimum_distance)
        self._shortest_distance = float(shortest_distance)
        self.curves = ["curve_a", "curve_b"]

    def shortest_distance(self):
        return self._shortest_distance


class _AlmFakeCurvatureObjective:
    def __init__(self, threshold, kappa_values, objective_value):
        self.threshold = float(threshold)
        self.curve = SimpleNamespace(
            kappa=lambda: np.asarray(kappa_values, dtype=float)
        )
        self._objective_value = float(objective_value)

    def J(self):
        return self._objective_value


class _AlmFakeDerivative:
    def __init__(self, gradient):
        self._gradient = np.asarray(gradient, dtype=float)

    def __call__(self, _objective):
        return self._gradient.copy()


class _AlmFakeCurrentObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def get_value(self):
        return self._value

    def vjp(self, value):
        cotangent = float(np.asarray(value, dtype=float).reshape(-1)[0])
        return _AlmFakeDerivative(cotangent * self._grad)


class _AlmFakeBiotSavart:
    def __init__(self, field_shape):
        self._field = np.zeros(field_shape, dtype=float)
        self.points = None

    def B(self):
        return self._field.copy()

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float).copy()

    def clear_cached_properties(self):
        pass


class _AlmFakeSurfaceNormals:
    def __init__(self, shape):
        self._unitnormal = np.zeros(shape, dtype=float)

    def unitnormal(self):
        return self._unitnormal.copy()

    def gamma(self):
        return np.zeros(self._unitnormal.shape, dtype=float)


def _alm_default_geometric_parity_kwargs():
    return dict(
        Jw=_AlmFakeWidthObjective(0.10, [0.0, 0.0]),
        width_min_threshold=0.05,
        width_max_threshold=0.17,
        Jself=_AlmFakeSelfIntersectObjective(
            0.0, [0.0, 0.0], shortest_self_distance=0.5
        ),
        self_intersect_threshold=0.0,
        length_min_target=0.95,
    )


def _alm_default_activity_tolerances():
    return lambda ds, cs: [
        ds * 4.0,
        cs * 4.0,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-6,
        1e-3,
        0.5,
    ]


def _alm_default_distance_constraint(*_args):
    return -0.008, np.array([0.6, 0.2]), -0.008


def _alm_default_curvature_constraint(*_args):
    return 0.75, np.array([0.9, -0.1])


def _build_untrusted_iota_state(objectives, handoff):
    """Construct a Stage2IotaState whose trust flags say 'untrusted but evaluable'.

    Uses the real ``compute_boozer_trust_state`` SSOT (so a regression that
    redefines ``iota_objective_active`` would also flip this fixture and the
    test pair) and the real ``_build_stage2_iota_state_from_iota`` helper to
    avoid fabricating the dataclass fields by hand.
    """
    boozer_surface = _make_ls_boozer_surface(
        jacobian=np.array([1e-3, 0.0]),  # ||grad||=1e-3 >> trust tol 1e-10
        iota=0.2,
    )
    trust = handoff.compute_boozer_trust_state(
        boozer_surface,
        solve_success=True,
        self_intersecting=False,
    )
    assert trust.boozer_trusted is False
    assert trust.iota_objective_active is False
    return objectives._build_stage2_iota_state_from_iota(
        0.2,
        target=0.2,
        tolerance=0.05,
        solve_failed=False,
        trust_state=trust,
    )


class TestStage2AlmProblemHonorsTrust:
    """Phase 1 trust gate: the ALM lane (``evaluate_stage2_alm_problem``)
    must zero the iota penalty signal and gradient when the Boozer trust
    gate flips ``iota_objective_active=False``, mirroring the soft-lane
    contract pinned by
    ``test_untrusted_but_evaluable_leaves_base_objective_unchanged``.

    Without these guards the ALM dual update ``λ_new = max(0, λ + ρ·c(x))``
    would feed a stale Boozer-domain residual ``c(x)`` into ``λ_iota`` while
    the iota gradient pushed the geometry around. The two assertions below
    pin the inversion such that any regression that re-injects a non-zero
    iota signal or a non-zero iota gradient would fail one of them.
    """

    def test_untrusted_iota_zeroes_signed_value_and_gradient(self):
        objectives = load_stage2_objectives_module()
        handoff = load_handoff_module()

        base_objective = _AlmFakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _AlmFakeSurfaceNormals((2, 2, 3))
        new_bs = _AlmFakeBiotSavart((4, 3))
        Jf = _AlmFakeScalarObjective(0.25)
        Jls = _AlmFakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _AlmFakeCurveDistance(0.05, 0.04)
        Jc = _AlmFakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _AlmFakeCurrentObjective(9500.0, [0.7, -0.4])
        # Penalty gradient that *would* push the optimizer if it were applied
        # — a regression that bypassed the trust gate would let this flow into
        # ``raw_constraint_grads[-1]`` and break the assertions below.
        live_penalty_grad = np.array([100.0, 200.0])

        # ``penalty_objective.dJ`` and ``iota_term.J`` should never be called
        # on the untrusted ALM path (the trust gate must short-circuit before
        # any Boozer-derived signal is consumed).
        stage2_iota_runtime = SimpleNamespace(
            mode="alm",
            target=0.2,
            tolerance=0.05,
            penalty_threshold=0.5,
            iota_term=SimpleNamespace(
                J=mock.Mock(
                    side_effect=AssertionError(
                        "iota_term.J() must not run when iota is untrusted"
                    )
                )
            ),
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError(
                        "penalty_objective.dJ() must not run when iota is untrusted"
                    )
                )
            ),
        )

        untrusted_state = _build_untrusted_iota_state(objectives, handoff)
        assert untrusted_state.iota_objective_active is False
        assert untrusted_state.boozer_trusted is False
        assert untrusted_state.solve_failed is False

        iota_evaluation = objectives.Stage2IotaEvaluation(
            state=untrusted_state,
            penalty_grad=live_penalty_grad,
        )

        with (
            mock.patch.object(
                objectives,
                "evaluate_stage2_iota",
                return_value=iota_evaluation,
            ),
            mock.patch("builtins.print"),
        ):
            result = objectives.evaluate_stage2_alm_problem(
                dofs=np.array([0.25, -0.4]),
                base_objective=base_objective,
                new_bs=new_bs,
                new_surf=new_surf,
                Jf=Jf,
                Jls=Jls,
                length_target=2.0,
                Jccdist=Jccdist,
                Jc=Jc,
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.zeros(9),
                penalty=12.0,
                stage2_constraint_activity_tolerances=(
                    _alm_default_activity_tolerances()
                ),
                smooth_min_distance_signed_constraint=(
                    _alm_default_distance_constraint
                ),
                smooth_max_curvature_signed_constraint=(
                    _alm_default_curvature_constraint
                ),
                stage2_iota_runtime=stage2_iota_runtime,
                **_alm_default_geometric_parity_kwargs(),
            )

        # The iota_penalty constraint is appended last
        # (see ``_stage2_constraint_names`` SSOT) so the last entry of every
        # per-constraint array is the iota term.
        assert result["constraint_names"][-1] == "iota_penalty"

        # (1) iota_signed_value == 0.0 — the dual-update signal used as
        # ``c(x)`` in ``λ + ρ·c(x)``.
        assert result["raw_dual_update_values"][-1] == pytest.approx(0.0)
        assert result["raw_hard_signed_constraint_values"][-1] == pytest.approx(0.0)

        # iota_violation == 0.0 — the feasibility signal used by ALM
        # convergence and outer-loop bookkeeping.
        assert result["raw_hard_violation_values"][-1] == pytest.approx(0.0)
        assert result["raw_feasibility_values"][-1] == pytest.approx(0.0)

        # (2) iota_grad is the zero vector — the gradient block in the
        # augmented Lagrangian that would otherwise push the geometry on an
        # untrusted Boozer signal.
        np.testing.assert_allclose(
            result["raw_constraint_grads"][-1],
            np.zeros_like(base_objective.dJ()),
        )

    def test_untrusted_iota_keeps_lambda_stable_across_outer_iteration(self):
        """Dual update ``λ_new = max(0, λ + ρ·signed_value)`` must not move
        the iota multiplier when the trust gate disables the iota lane.

        Mirrors ``test_untrusted_but_evaluable_leaves_base_objective_unchanged``
        in spirit: the optimizer state (here, the ALM dual variable) is
        invariant under an untrusted iteration. Without the zero-signal
        guard at stage2_objectives.py:2245-2247, any nonzero
        ``signed_value`` would shift ``λ_iota`` and induce
        dual-multiplier oscillation as trust toggled between outer
        iterations.
        """
        objectives = load_stage2_objectives_module()
        handoff = load_handoff_module()

        base_objective = _AlmFakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _AlmFakeSurfaceNormals((2, 2, 3))
        new_bs = _AlmFakeBiotSavart((4, 3))
        Jf = _AlmFakeScalarObjective(0.25)
        Jls = _AlmFakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _AlmFakeCurveDistance(0.05, 0.04)
        Jc = _AlmFakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _AlmFakeCurrentObjective(9500.0, [0.7, -0.4])

        stage2_iota_runtime = SimpleNamespace(
            mode="alm",
            target=0.2,
            tolerance=0.05,
            penalty_threshold=0.5,
            iota_term=SimpleNamespace(J=lambda: 0.2),
            penalty_objective=SimpleNamespace(dJ=lambda: np.zeros(2)),
        )
        untrusted_state = _build_untrusted_iota_state(objectives, handoff)
        iota_evaluation = objectives.Stage2IotaEvaluation(
            state=untrusted_state,
            penalty_grad=np.array([100.0, 200.0]),  # would-be live signal
        )

        # Pre-iteration multipliers — give λ_iota a deliberately non-trivial
        # positive value so that any drift across the outer iteration would
        # be measurable. The geometric constraints get small λ values so
        # this test stays focused on the iota lane.
        pre_lambda_iota = 0.75
        multipliers = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, pre_lambda_iota]
        )
        penalty = 12.0

        with (
            mock.patch.object(
                objectives,
                "evaluate_stage2_iota",
                return_value=iota_evaluation,
            ),
            mock.patch("builtins.print"),
        ):
            result = objectives.evaluate_stage2_alm_problem(
                dofs=np.array([0.25, -0.4]),
                base_objective=base_objective,
                new_bs=new_bs,
                new_surf=new_surf,
                Jf=Jf,
                Jls=Jls,
                length_target=2.0,
                Jccdist=Jccdist,
                Jc=Jc,
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=multipliers,
                penalty=penalty,
                stage2_constraint_activity_tolerances=(
                    _alm_default_activity_tolerances()
                ),
                smooth_min_distance_signed_constraint=(
                    _alm_default_distance_constraint
                ),
                smooth_max_curvature_signed_constraint=(
                    _alm_default_curvature_constraint
                ),
                stage2_iota_runtime=stage2_iota_runtime,
                **_alm_default_geometric_parity_kwargs(),
            )

        # Drive the real ALM dual-update SSOT — same function the outer
        # solver calls between iterations — using the freshly-computed
        # ``raw_dual_update_values``.
        assert result["constraint_names"][-1] == "iota_penalty"
        updated_multipliers = _updated_nonnegative_multipliers(
            multipliers,
            np.asarray(result["raw_dual_update_values"], dtype=float),
            penalty,
        )

        # λ_iota must be identical across the untrusted iteration.
        assert updated_multipliers[-1] == pytest.approx(pre_lambda_iota)
