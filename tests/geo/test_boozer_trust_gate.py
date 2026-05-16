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

import numpy as np
import pytest


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


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

    def test_ls_path_returns_none_when_jacobian_missing(self):
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
