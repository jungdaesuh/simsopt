"""Contract for the target-lane Boozer Newton-polish policy resolution.

The single-stage path decides, per lane, whether the JAX ``ondevice`` Boozer
dense Newton polish runs. It previously defaulted to ``"skip-large-strict-cuda"``,
which made the two parity lanes **asymmetric** at large resolution (mpol/ntor >= 6):
the GPU target lane resolved to ``"skip"`` (a BFGS-only surface, ~1e-4 Boozer
residual), while the JAX-CPU reference lane resolved to ``None`` -- the parent
dropped the override because it computed the effective inner-Boozer backend from
the *outer* ``"scipy-jax"`` value (a CUDA-only approximation) instead of the routed
``"ondevice"`` value, so the CPU child fell back to its default and *ran* the full
Newton. Parity therefore compared a BFGS-only GPU surface against a
Newton-converged CPU surface (apples-to-oranges), and the CPU Newton was the slow
lane. Verified against the parent commit's resolver.

The policy is now an explicit ``"run"``/``"skip"`` choice that defaults to
``"run"`` on every platform -- matching the production GPU lane, which already
passes ``--target-lane-boozer-newton-polish-policy run`` and converges the inner
ondevice Newton via the strict-clean traceable path. These tests pin that
contract at both resolvers: the benchmark parent (which builds the child command)
and the example child (which maps the policy onto the BoozerSurfaceJAX option).
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from benchmarks.single_stage_init_parity import (
    _resolve_target_lane_boozer_newton_polish_policy,
    parse_args,
)
from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    build_target_lane_trial_boozer_overrides,
    resolve_effective_boozer_newton_polish_policy_override,
    resolve_effective_trial_boozer_newton_polish_policy_override,
)


def _parse(extra: list[str]):
    argv = [
        "single_stage_init_parity.py",
        "--output-json",
        "/tmp/ss_out.json",
        "--platform",
        "cuda",
        "--mpol",
        "10",
        "--ntor",
        "10",
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


# --- benchmark parent resolver (decides the child command flag) ---


def test_cuda_large_resolution_defaults_to_run():
    """The GPU lane no longer silently skips the Boozer Newton polish.

    mpol/ntor = 10 (>= the old skip threshold of 6) on cuda with the default
    policy must resolve to ``"run"``; previously this was ``"skip"``.
    """
    args = _parse([])
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="jax", platform="cuda", args=args
    )
    assert policy == "run", policy


def test_cuda_parent_jax_cpu_reference_child_defaults_to_run():
    """The JAX CPU reference child no longer inherits a CUDA-parent skip."""
    args = _parse([])
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="jax", platform="cpu", args=args
    )
    assert policy == "run", policy


def test_explicit_skip_is_honored():
    """An explicit ``--target-lane-boozer-newton-polish-policy skip`` still skips."""
    args = _parse(["--target-lane-boozer-newton-polish-policy", "skip"])
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="jax", platform="cuda", args=args
    )
    assert policy == "skip", policy


def test_explicit_skip_is_honored_for_target_lane_cpu_child():
    """The default target-lane CPU child still has an ondevice Boozer solve."""
    args = _parse(["--target-lane-boozer-newton-polish-policy", "skip"])
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="jax", platform="cpu", args=args
    )
    assert policy == "skip", policy


def test_scipy_jax_fullgraph_child_gets_no_ondevice_policy_override():
    """Fullgraph's scipy Boozer child does not receive a JAX/ondevice override."""
    args = _parse(
        [
            "--optimizer-backend",
            "scipy-jax-fullgraph",
            "--target-lane-boozer-newton-polish-policy",
            "skip",
        ]
    )
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="jax", platform="cpu", args=args
    )
    assert policy is None, policy


def test_non_jax_reference_lane_gets_no_override():
    """The cpp/CPU reference lane never carries a JAX-only Boozer override."""
    args = _parse([])
    policy = _resolve_target_lane_boozer_newton_polish_policy(
        backend="cpp", platform="cpu", args=args
    )
    assert policy is None, policy


def test_legacy_skip_large_strict_cuda_token_is_rejected():
    """The removed ``skip-large-strict-cuda`` token is no longer a valid choice."""
    argv = [
        "single_stage_init_parity.py",
        "--output-json",
        "/tmp/ss_out.json",
        "--platform",
        "cuda",
        "--target-lane-boozer-newton-polish-policy",
        "skip-large-strict-cuda",
    ]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        parse_args()


# --- example child resolver (maps the policy onto the adapter option) ---


def test_child_override_defaults_to_run_on_jax_ondevice():
    """The child's adapter override defaults to ``"run"`` for a JAX ondevice solve.

    Previously, on a large strict-CUDA child, this returned ``"skip"``.
    """
    override = resolve_effective_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_boozer_newton_polish_policy=None,
    )
    assert override == "run", override


def test_child_override_honors_explicit_skip():
    override = resolve_effective_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_boozer_newton_polish_policy="skip",
    )
    assert override == "skip", override


def test_child_override_is_none_off_the_jax_ondevice_path():
    """No override for non-JAX or non-ondevice Boozer solves (adapter keeps its default)."""
    assert (
        resolve_effective_boozer_newton_polish_policy_override(
            field_backend="cpp",
            optimizer_backend="ondevice",
            target_lane_boozer_newton_polish_policy=None,
        )
        is None
    )
    assert (
        resolve_effective_boozer_newton_polish_policy_override(
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_optimizer_backend="scipy",
            target_lane_boozer_newton_polish_policy=None,
        )
        is None
    )


def test_child_trial_override_defaults_to_skip_on_jax_ondevice():
    """Trial solves default to bounded K1 work; full-fidelity solves still run."""
    full_override = resolve_effective_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_boozer_newton_polish_policy=None,
    )
    trial_override = resolve_effective_trial_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_trial_boozer_newton_polish_policy=None,
    )

    assert full_override == "run"
    assert trial_override == "skip"


def test_child_trial_override_honors_explicit_run():
    override = resolve_effective_trial_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_trial_boozer_newton_polish_policy="run",
    )
    assert override == "run", override


def test_trial_boozer_overrides_use_trial_policy_not_full_policy():
    """The temporary trial solve override is independent of full solve fidelity."""
    full_override = resolve_effective_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_boozer_newton_polish_policy=None,
    )
    trial_override = resolve_effective_trial_boozer_newton_polish_policy_override(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_trial_boozer_newton_polish_policy=None,
    )
    overrides = build_target_lane_trial_boozer_overrides(
        bfgs_tol=None,
        bfgs_maxiter=None,
        newton_tol=None,
        newton_maxiter=None,
        newton_stab=None,
        newton_polish_policy=trial_override,
    )

    assert full_override == "run"
    assert overrides["newton_polish_policy"] == "skip"
