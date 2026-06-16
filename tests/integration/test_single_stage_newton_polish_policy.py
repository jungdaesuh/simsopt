"""Contract for the target-lane Boozer Newton-polish policy resolution.

The single-stage path decides, per lane, whether the JAX ``ondevice`` Boozer
dense Newton polish runs. It previously defaulted to ``"skip-large-strict-cuda"``.
The parent resolver keyed that skip on the *parent* ``--platform``, so on a large
strict-CUDA parity run (mpol/ntor >= 6) it returned ``"skip"`` for **both** the
GPU target child and its JAX-CPU reference child. Parity therefore compared two
least-squares-only surfaces (~1e-4 Boozer residual) rather than two fully
Newton-converged surfaces, and the silent skip was platform-dependent.

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
    resolve_effective_boozer_newton_polish_policy_override,
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


def test_explicit_skip_is_honored_for_scipy_jax_cpu_child():
    """The default scipy-jax CPU child still has an ondevice Boozer solve."""
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
