"""Regression tests for the CPU ``--xla_cpu_opt_preset=FAST_COMPILE`` knob.

Plan reference: ``docs/single_stage_compile_blowup_fix_implementation_plan.md``
Phase 2. ``apply_jax_runtime_config`` pulls the FAST_COMPILE CPU compile preset
into ``XLA_FLAGS`` before JAX initializes (matching TORAX's
``torax/__init__.py`` behavior), because XLA reads ``XLA_FLAGS`` only at backend
initialization. These tests pin the two contracts that matter:

- ``_xla_flags_with_cpu_compile_preset`` composes the preset non-destructively
  and idempotently, and never overrides a caller-supplied preset.
- ``_apply_cpu_compile_preset_env`` applies it on non-parity CPU lanes only (the
  CUDA backend carries the determinism contract in ``XLA_FLAGS`` instead, and
  bit-exact parity lanes keep their existing XLA pass set).
"""

from __future__ import annotations

import os
import types

import pytest

from simsopt_jax.backend.runtime import (
    _CPU_OPT_PRESET_FAST_COMPILE,
    _XLA_FLAGS_ENV,
    _apply_cpu_compile_preset_env,
    _xla_flags_with_cpu_compile_preset,
)


# ---------------------------------------------------------------------------
# Pure composition: _xla_flags_with_cpu_compile_preset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_compose_yields_lone_preset_for_empty_input(empty):
    """No prior flags must produce exactly the preset token (no stray space)."""
    assert _xla_flags_with_cpu_compile_preset(empty) == _CPU_OPT_PRESET_FAST_COMPILE


def test_compose_preserves_existing_unrelated_flags():
    """An unrelated flag is kept verbatim with the preset appended after it."""
    existing = "--xla_force_host_platform_device_count=8"
    assert _xla_flags_with_cpu_compile_preset(existing) == (
        f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"
    )


def test_compose_is_idempotent_for_same_preset():
    """Re-composing an already-present FAST_COMPILE preset must not duplicate it."""
    once = _xla_flags_with_cpu_compile_preset(None)
    assert _xla_flags_with_cpu_compile_preset(once) == once


@pytest.mark.parametrize(
    "existing",
    [
        "--xla_cpu_opt_preset",  # bare flag, no value (exercises the `==` guard arm)
        "--xla_cpu_opt_preset=DEFAULT",
        "--a=1 --xla_cpu_opt_preset=DEFAULT --b=2",
    ],
)
def test_compose_respects_caller_supplied_preset(existing):
    """A caller's explicit preset (any value) wins; we neither override nor dup."""
    assert _xla_flags_with_cpu_compile_preset(existing) == existing


def test_compose_does_not_match_prefix_lookalike_flag():
    """A flag that merely starts with the preset name must not block appending."""
    existing = "--xla_cpu_opt_preset_extra=1"
    assert _xla_flags_with_cpu_compile_preset(existing) == (
        f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"
    )


@pytest.mark.parametrize("existing", ['--a="unterminated', "--a='unterminated"])
def test_compose_preserves_malformed_user_flags(existing):
    """Malformed user flags are still preserved; the helper must not overwrite them."""
    assert _xla_flags_with_cpu_compile_preset(existing) == (
        f"{existing} {_CPU_OPT_PRESET_FAST_COMPILE}"
    )


# ---------------------------------------------------------------------------
# CPU-scoped application: _apply_cpu_compile_preset_env
# ---------------------------------------------------------------------------


def _config(jax_platform):
    return types.SimpleNamespace(jax_platform=jax_platform)


def _policy(parity_mode=False):
    return types.SimpleNamespace(parity_mode=parity_mode)


def test_apply_is_noop_on_cuda(monkeypatch):
    """CUDA must not gain the CPU preset; its XLA_FLAGS stay untouched."""
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cpu_compile_preset_env(_config("cuda"), _policy())
    assert _XLA_FLAGS_ENV not in os.environ


def test_apply_is_noop_on_cpu_parity(monkeypatch):
    """The bit-exact CPU parity lane must not gain the optimization-skipping preset."""
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cpu_compile_preset_env(_config("cpu"), _policy(parity_mode=True))
    assert _XLA_FLAGS_ENV not in os.environ


def test_apply_sets_preset_on_cpu(monkeypatch):
    """A non-parity CPU lane with no prior flags gets exactly the preset."""
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cpu_compile_preset_env(_config("cpu"), _policy(parity_mode=False))
    assert os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE


def test_apply_is_idempotent_across_repeated_calls(monkeypatch):
    """Re-applying the config (e.g. repeated init) must not duplicate the preset."""
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    _apply_cpu_compile_preset_env(_config("cpu"), _policy())
    _apply_cpu_compile_preset_env(_config("cpu"), _policy())
    assert os.environ[_XLA_FLAGS_ENV] == _CPU_OPT_PRESET_FAST_COMPILE
