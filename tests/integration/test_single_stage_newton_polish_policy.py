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

The full-fidelity policy is now an explicit ``"run"``/``"skip"`` choice that
defaults to ``"run"`` on every platform -- matching the production GPU lane,
which already passes ``--target-lane-boozer-newton-polish-policy run`` and
converges the inner ondevice Newton via the strict-clean traceable path. The
trial-only policy is resolved separately and defaults to ``"skip"`` for
line-search probes. These tests pin both contracts at the benchmark parent
(which builds the child command) and the example child (which maps the policies
onto the BoozerSurfaceJAX options).
"""

from __future__ import annotations

from collections import namedtuple
import sys
from unittest import mock

import numpy as np
import pytest

from simsopt_jax_adapters.geo.surface_objectives import _TRACEABLE_RUNTIME_OPTION_KEYS

from benchmarks.single_stage_init_parity import (
    _append_optional_single_stage_flags,
    _resolve_target_lane_boozer_newton_polish_policy,
    _resolve_target_lane_trial_boozer_newton_polish_policy,
    _single_stage_full_run_family_id,
    parse_args,
)
from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    _build_decomposed_coil_host_value_and_grad,
    _cached_target_lane_objective_evaluation_sync_state,
    build_target_lane_full_state_value_and_grad,
    build_single_stage_target_lane_objective_evaluation_trace_wrapper,
    build_target_lane_trial_boozer_solver_trace_metadata,
    build_target_lane_trial_boozer_overrides,
    resolve_effective_boozer_newton_polish_policy_override,
    resolve_effective_trial_boozer_newton_polish_policy_override,
    resolve_target_lane_trial_boozer_bfgs_maxiter,
    resolve_target_lane_trial_boozer_bfgs_tol,
    wrap_target_lane_solved_pair_with_boozer_overrides,
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


def test_parent_trial_policy_defaults_to_skip_for_target_lane_cuda_child():
    args = _parse([])
    policy = _resolve_target_lane_trial_boozer_newton_polish_policy(
        backend="jax",
        platform="cuda",
        args=args,
    )
    assert policy == "skip", policy


def test_parent_trial_policy_honors_explicit_run():
    args = _parse(["--target-lane-trial-boozer-newton-polish-policy", "run"])
    policy = _resolve_target_lane_trial_boozer_newton_polish_policy(
        backend="jax",
        platform="cuda",
        args=args,
    )
    assert policy == "run", policy


def test_parent_accepts_trial_bfgs_budget_flags():
    args = _parse(
        [
            "--target-lane-trial-boozer-bfgs-tol",
            "1e-7",
            "--target-lane-trial-boozer-bfgs-maxiter",
            "25",
        ]
    )

    assert args.target_lane_trial_boozer_bfgs_tol == 1e-7
    assert args.target_lane_trial_boozer_bfgs_maxiter == 25


def test_parent_forwards_trial_bfgs_budget_to_child_command():
    command: list[str] = []

    _append_optional_single_stage_flags(
        command,
        benchmark_mode=False,
        minimal_artifacts=False,
        profile_target_lane=False,
        profile_target_lane_only=False,
        diagnose_target_lane_scaled_phase1=False,
        record_target_lane_invalid_state_events=False,
        profile_target_lane_batch_size=None,
        enable_compile_diagnostics=False,
        jax_profile_dir=None,
        experimental_target_lane_value_and_grad=False,
        disable_target_lane_success_filter=False,
        record_objective_evaluation_trace=False,
        record_target_optimizer_state_trace=False,
        target_lane_trial_boozer_bfgs_tol=1e-7,
        target_lane_trial_boozer_bfgs_maxiter=25,
    )

    assert "--target-lane-trial-boozer-bfgs-tol" in command
    assert command[
        command.index("--target-lane-trial-boozer-bfgs-tol") + 1
    ] == "1e-07"
    assert "--target-lane-trial-boozer-bfgs-maxiter" in command
    assert command[
        command.index("--target-lane-trial-boozer-bfgs-maxiter") + 1
    ] == "25"


def test_run_family_id_includes_full_and_trial_bfgs_budget():
    base_args = _parse(
        [
            "--target-lane-boozer-bfgs-maxiter",
            "1500",
            "--target-lane-trial-boozer-bfgs-maxiter",
            "25",
        ]
    )
    changed_full_args = _parse(
        [
            "--target-lane-boozer-bfgs-maxiter",
            "300",
            "--target-lane-trial-boozer-bfgs-maxiter",
            "25",
        ]
    )
    changed_trial_args = _parse(
        [
            "--target-lane-boozer-bfgs-maxiter",
            "1500",
            "--target-lane-trial-boozer-bfgs-maxiter",
            "50",
        ]
    )

    base_id = _single_stage_full_run_family_id(
        base_args,
        runtime_seed_spec_hash="seed",
        objective_configuration_hash="objective",
    )
    changed_full_id = _single_stage_full_run_family_id(
        changed_full_args,
        runtime_seed_spec_hash="seed",
        objective_configuration_hash="objective",
    )
    changed_trial_id = _single_stage_full_run_family_id(
        changed_trial_args,
        runtime_seed_spec_hash="seed",
        objective_configuration_hash="objective",
    )

    assert base_id != changed_full_id
    assert base_id != changed_trial_id


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
    trial_policy = _resolve_target_lane_trial_boozer_newton_polish_policy(
        backend="jax",
        platform="cpu",
        args=args,
    )
    assert trial_policy is None, trial_policy


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


def test_child_trial_bfgs_overrides_validate_explicit_budget():
    tol = resolve_target_lane_trial_boozer_bfgs_tol(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_trial_boozer_bfgs_tol=1e-7,
    )
    maxiter = resolve_target_lane_trial_boozer_bfgs_maxiter(
        field_backend="jax",
        optimizer_backend="ondevice",
        target_lane_trial_boozer_bfgs_maxiter=25,
    )

    assert tol == 1e-7
    assert maxiter == 25


def test_child_trial_bfgs_overrides_reject_invalid_budget():
    with pytest.raises(ValueError, match="target_lane_trial_boozer_bfgs_tol"):
        resolve_target_lane_trial_boozer_bfgs_tol(
            field_backend="jax",
            optimizer_backend="ondevice",
            target_lane_trial_boozer_bfgs_tol=0.0,
        )

    with pytest.raises(ValueError, match="target_lane_trial_boozer_bfgs_maxiter"):
        resolve_target_lane_trial_boozer_bfgs_maxiter(
            field_backend="jax",
            optimizer_backend="ondevice",
            target_lane_trial_boozer_bfgs_maxiter=0,
        )


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


def test_trial_boozer_overrides_carry_trial_bfgs_budget():
    overrides = build_target_lane_trial_boozer_overrides(
        bfgs_tol=1e-6,
        bfgs_maxiter=12,
        newton_tol=None,
        newton_maxiter=None,
        newton_stab=None,
        newton_polish_policy="skip",
    )

    assert overrides["bfgs_tol"] == 1e-6
    assert overrides["bfgs_maxiter"] == 12
    assert overrides["newton_polish_policy"] == "skip"


def test_trial_boozer_metadata_reports_trial_bfgs_budget():
    class BoozerSurface:
        def __init__(self):
            self.res = {"type": "ls"}
            self.options = {
                "bfgs_tol": 1e-12,
                "bfgs_maxiter": 1500,
                "newton_polish_policy": "run",
            }

    metadata = build_target_lane_trial_boozer_solver_trace_metadata(
        BoozerSurface(),
        {
            "bfgs_tol": 1e-6,
            "bfgs_maxiter": 12,
            "newton_polish_policy": "skip",
        },
    )

    assert metadata["bfgs_tol"] == 1e-6
    assert metadata["bfgs_maxiter"] == 12
    assert metadata["newton_polish_policy"] == "skip"


def test_trial_solve_cache_keeps_trace_reuse_but_disables_final_sync_reuse():
    solved_pair_type = namedtuple(
        "SolvedPair", ["solve_fn", "value_grad_from_solved"]
    )
    coil_dofs = [1.0, 2.0]
    forward_result = {
        "success": True,
        "primal_success": True,
        "sdofs": [3.0, 4.0],
        "iota": 0.11,
        "G": 2.0,
        "x": [5.0, 6.0],
        "linear_solve_factors": {},
    }

    def solve_fn(_coil_dofs):
        return dict(forward_result)

    def value_grad_from_solved(_coil_dofs, _x, _linear_solve_factors):
        return 7.0, [0.1, 0.2]

    objective = _build_decomposed_coil_host_value_and_grad(
        solved_pair_type(
            solve_fn=solve_fn,
            value_grad_from_solved=value_grad_from_solved,
        ),
        baseline_gradient=[0.0, 0.0],
        cache_solve_result_for_reporting=False,
    )

    value, grad = objective(coil_dofs)

    assert float(np.asarray(value)) == 7.0
    np.testing.assert_allclose(np.asarray(grad), [0.1, 0.2])
    assert objective.cached_forward_result_for_coil_dofs(coil_dofs)["iota"] == 0.11
    assert objective.target_lane_solve_result_for_coil_dofs(coil_dofs) is None


def test_trial_trace_wrapper_reuses_k1_without_final_sync_cache():
    solved_pair_type = namedtuple(
        "SolvedPair", ["solve_fn", "value_grad_from_solved"]
    )
    optimizer_dofs = np.array([1.0, 2.0])
    forward_result = {
        "success": True,
        "primal_success": True,
        "sdofs": [3.0, 4.0],
        "iota": 0.11,
        "G": 2.0,
        "x": [5.0, 6.0],
        "linear_solve_factors": {},
    }

    def solve_fn(_coil_dofs):
        return dict(forward_result)

    def value_grad_from_solved(_coil_dofs, _x, _linear_solve_factors):
        return 7.0, [0.1, 0.2]

    objective = _build_decomposed_coil_host_value_and_grad(
        solved_pair_type(
            solve_fn=solve_fn,
            value_grad_from_solved=value_grad_from_solved,
        ),
        baseline_gradient=[0.0, 0.0],
        cache_solve_result_for_reporting=False,
    )

    def unexpected_forward_result(_coil_dofs):
        raise AssertionError("trace wrapper should reuse the decomposed K1 result")

    run_dict = {}
    traced_objective = build_single_stage_target_lane_objective_evaluation_trace_wrapper(
        target_value_and_grad_objective=objective,
        target_forward_result=unexpected_forward_result,
        optimizer_to_coil_dofs=lambda x: x,
        run_dict=run_dict,
    )

    value, grad = traced_objective(optimizer_dofs)

    assert float(np.asarray(value)) == 7.0
    np.testing.assert_allclose(np.asarray(grad), [0.1, 0.2])
    assert (
        _cached_target_lane_objective_evaluation_sync_state(
            run_dict,
            optimizer_dofs,
        )
        is None
    )


def test_full_state_wrapper_preserves_trial_reporting_cache_contract():
    def value_and_grad_objective(_coil_dofs):
        return 1.0, np.array([0.1, 0.2])

    value_and_grad_objective.cache_solve_result_for_reporting = False

    class OptimizerDofMap:
        def jax_coil_dofs_from_optimizer_dofs(self, optimizer_dofs):
            return optimizer_dofs

        def jax_full_gradient_from_coil_gradient(
            self,
            _optimizer_dofs,
            coil_gradient,
        ):
            return coil_gradient

    full_state_objective = build_target_lane_full_state_value_and_grad(
        value_and_grad_objective,
        OptimizerDofMap(),
    )

    assert full_state_objective.cache_solve_result_for_reporting is False


def test_traceable_runtime_cache_key_includes_newton_polish_policy():
    """Changing full vs trial Newton polish policy must rebuild the K1 runtime."""
    assert "newton_polish_policy" in _TRACEABLE_RUNTIME_OPTION_KEYS


def test_decomposed_solved_pair_applies_trial_policy_at_solve_call_time():
    """SciPy calls the split K1 solve after construction-time overrides restore."""
    solved_pair_type = namedtuple(
        "SolvedPair", ["solve_fn", "value_grad_from_solved"]
    )
    observed_policies = []

    class BoozerSurface:
        def __init__(self):
            self.options = {
                "newton_polish_policy": "run",
                "newton_stab": 0.0,
            }

    boozer_surface = BoozerSurface()

    def solve_fn(coil_dofs):
        observed_policies.append(boozer_surface.options["newton_polish_policy"])
        return {"coil_dofs": coil_dofs}

    solved_pair = solved_pair_type(
        solve_fn=solve_fn,
        value_grad_from_solved=object(),
    )
    wrapped_pair = wrap_target_lane_solved_pair_with_boozer_overrides(
        boozer_surface,
        solved_pair,
        {"newton_polish_policy": "skip"},
    )

    assert boozer_surface.options["newton_polish_policy"] == "run"
    assert wrapped_pair.solve_fn("candidate") == {"coil_dofs": "candidate"}
    assert observed_policies == ["skip"]
    assert boozer_surface.options["newton_polish_policy"] == "run"
