"""Regression tests for the CUDA ``--xla_gpu_experimental_enable_fusion_autotuner=false`` knob.

``apply_jax_runtime_config`` pulls the flag into ``XLA_FLAGS`` before JAX
initializes on CUDA lanes, because on the RTX 5090 with jaxlib 0.10.0
(2026-09-13) XLA's experimental fusion autotuner faulted inside most fresh
compiles of the GPMO ``ArbVec_backtracking`` program with
``CUDA_ERROR_ILLEGAL_ADDRESS`` (interleaved 4 of 6 with it on, 0 of 6 off);
the suspected, unproven cause is its Triton block-level candidates running
data-indexed fusions on autotuning buffers. These tests pin the two contracts
that matter:

- ``_xla_flags_with_gpu_fusion_autotuner_disabled`` composes the flag
  non-destructively and idempotently, and never overrides a caller-supplied value.
- ``_apply_cuda_fusion_autotuner_env`` applies it on CUDA lanes only.
"""

from __future__ import annotations

import os
import types

import pytest
from simsopt_jax.backend.runtime import (
    _CPU_OPT_PRESET_FAST_COMPILE,
    _GPU_FUSION_AUTOTUNER_DISABLED,
    _XLA_FLAGS_ENV,
    _apply_cuda_fusion_autotuner_env,
    _xla_flags_with_cpu_compile_preset,
    _xla_flags_with_gpu_fusion_autotuner_disabled,
)


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_compose_yields_lone_flag_for_empty_input(empty):
    assert (
        _xla_flags_with_gpu_fusion_autotuner_disabled(empty)
        == _GPU_FUSION_AUTOTUNER_DISABLED
    )


def test_compose_preserves_existing_unrelated_flags():
    existing = "--xla_gpu_exclude_nondeterministic_ops=true"
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == (
        f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_compose_is_idempotent():
    once = _xla_flags_with_gpu_fusion_autotuner_disabled(None)
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(once) == once


@pytest.mark.parametrize(
    "existing",
    [
        "--xla_gpu_experimental_enable_fusion_autotuner",
        "--xla_gpu_experimental_enable_fusion_autotuner=true",
        "--a=1 --xla_gpu_experimental_enable_fusion_autotuner=true --b=2",
    ],
)
def test_compose_respects_caller_supplied_value(existing):
    """A caller who deliberately keeps the autotuner on is not overridden."""
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == existing


def test_compose_does_not_match_prefix_lookalike_flag():
    existing = "--xla_gpu_experimental_enable_fusion_autotuner_extra=1"
    assert _xla_flags_with_gpu_fusion_autotuner_disabled(existing) == (
        f"{existing} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_compose_composes_with_the_cpu_preset_helper():
    """Both helpers share one composition rule, so they stack in either order."""
    both = _xla_flags_with_gpu_fusion_autotuner_disabled(
        _xla_flags_with_cpu_compile_preset(None)
    )
    assert both == f"{_CPU_OPT_PRESET_FAST_COMPILE} {_GPU_FUSION_AUTOTUNER_DISABLED}"
    assert _xla_flags_with_cpu_compile_preset(both) == both


def _config(jax_platform):
    return types.SimpleNamespace(jax_platform=jax_platform)


def test_apply_is_noop_on_cpu(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_fusion_autotuner_env(_config("cpu"))
    assert _XLA_FLAGS_ENV not in os.environ


def test_apply_sets_flag_on_cuda(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_fusion_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == _GPU_FUSION_AUTOTUNER_DISABLED


def test_apply_keeps_caller_flags_on_cuda(monkeypatch):
    monkeypatch.setenv(_XLA_FLAGS_ENV, "--xla_gpu_exclude_nondeterministic_ops=true")
    _apply_cuda_fusion_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == (
        f"--xla_gpu_exclude_nondeterministic_ops=true {_GPU_FUSION_AUTOTUNER_DISABLED}"
    )


def test_apply_is_idempotent_across_repeated_calls(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cuda_fusion_autotuner_env(_config("cuda"))
    _apply_cuda_fusion_autotuner_env(_config("cuda"))
    assert os.environ[_XLA_FLAGS_ENV] == _GPU_FUSION_AUTOTUNER_DISABLED
