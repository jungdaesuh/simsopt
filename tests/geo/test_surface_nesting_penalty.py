"""Tests for the soft surface-nesting penalty barrier.

The load-bearing check is the one-sided quadratic-barrier semantics: the penalty
must be ``0.5*weight*max(0, signed_value)**2`` with the chain-rule gradient
``weight*max(0, signed_value)*signed_grad``, ACTIVE only inside the clearance band
(``signed_value > 0``) and inert outside it (including ``weight <= 0``). This is the
weighted-sum-mode equivalent of the augmented-Lagrangian surface-stack spacing
constraint; both consume the SAME ``smooth_min_surface_stack_signed_constraint``
margin, so these tests pin only the barrier wrapper, not the (separately-tested,
coil-live) margin itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_constraints import (  # noqa: E402
    surface_stack_nesting_penalty,
)


def test_active_band_quadratic_value_and_chain_rule_gradient():
    # signed_value > 0 (adjacent surfaces inside the clearance band): the barrier is
    # 0.5*w*c^2 and the gradient is w*c*signed_grad (chain rule through the margin).
    signed_grad = np.array([1.0, -2.0, 0.5])
    value, grad = surface_stack_nesting_penalty(2.0, signed_grad, weight=3.0)
    assert value == pytest.approx(0.5 * 3.0 * 2.0**2)  # 6.0
    np.testing.assert_allclose(grad, 3.0 * 2.0 * signed_grad)  # [6, -12, 1.5]


def test_inert_outside_band_negative_margin():
    # signed_value < 0 (comfortably spaced): the one-sided barrier is exactly zero in
    # both value and gradient, no matter how large the weight is.
    signed_grad = np.array([1.0, -1.0])
    value, grad = surface_stack_nesting_penalty(-0.3, signed_grad, weight=1000.0)
    assert value == 0.0
    np.testing.assert_array_equal(grad, np.zeros_like(signed_grad))


def test_boundary_margin_is_zero():
    # signed_value == 0 (exactly at the clearance knee): zero value and gradient.
    signed_grad = np.array([2.0, 2.0])
    value, grad = surface_stack_nesting_penalty(0.0, signed_grad, weight=5.0)
    assert value == 0.0
    np.testing.assert_array_equal(grad, np.zeros_like(signed_grad))


def test_zero_weight_is_inert_default():
    # weight == 0 (the default-OFF lever): zero value and gradient even with a large
    # active margin -- the byte-identical no-op that keeps default runs unchanged.
    signed_grad = np.array([3.0, -4.0])
    value, grad = surface_stack_nesting_penalty(5.0, signed_grad, weight=0.0)
    assert value == 0.0
    np.testing.assert_array_equal(grad, np.zeros_like(signed_grad))


def test_value_scales_quadratically_grad_linearly_in_band():
    # Doubling the in-band margin quadruples the penalty value and doubles the
    # gradient magnitude -- the barrier is quadratic in the margin.
    signed_grad = np.array([1.0, 0.0, -1.0])
    v1, g1 = surface_stack_nesting_penalty(1.0, signed_grad, weight=2.0)
    v2, g2 = surface_stack_nesting_penalty(2.0, signed_grad, weight=2.0)
    assert v2 == pytest.approx(4.0 * v1)
    np.testing.assert_allclose(g2, 2.0 * g1)


def test_gradient_is_linear_in_weight():
    # Gradient magnitude scales linearly with the weight, so the lever's strength is
    # exactly the weight knob.
    signed_grad = np.array([0.5, -0.5])
    _, g1 = surface_stack_nesting_penalty(2.0, signed_grad, weight=1.0)
    _, g10 = surface_stack_nesting_penalty(2.0, signed_grad, weight=10.0)
    np.testing.assert_allclose(g10, 10.0 * g1)


def test_finite_difference_consistency_in_band():
    # (value, grad) are a consistent (f, df) pair w.r.t. the margin c: a central
    # finite difference of the value reproduces the analytic d/dc [0.5*w*max(0,c)^2]
    # = w*c, and the returned gradient is that slope times signed_grad. Catches a
    # regression in the 0.5 coefficient or the chain rule.
    signed_grad = np.array([1.0, -2.0, 0.5])
    weight = 3.0
    c0 = 1.5
    eps = 1e-7
    v_plus, _ = surface_stack_nesting_penalty(c0 + eps, signed_grad, weight)
    v_minus, _ = surface_stack_nesting_penalty(c0 - eps, signed_grad, weight)
    dvalue_dc_fd = (v_plus - v_minus) / (2 * eps)
    dvalue_dc_analytic = weight * c0
    assert dvalue_dc_fd == pytest.approx(dvalue_dc_analytic, rel=1e-5)
    _, grad = surface_stack_nesting_penalty(c0, signed_grad, weight)
    np.testing.assert_allclose(grad, dvalue_dc_analytic * signed_grad, rtol=1e-12)
