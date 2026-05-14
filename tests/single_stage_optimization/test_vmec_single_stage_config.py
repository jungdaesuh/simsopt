"""Pure unit tests for ``VmecSingleStageConfig``.

The config dataclass is the SSOT for VMEC single-stage runs (see plan
section "Frozen configuration"). Tests pin:

- the frozen-dataclass contract (mutation raises ``FrozenInstanceError``).
- the bring-up placeholder values that participate in the artifact-manifest
  ``config_hash`` (plan section "Artifact Manifest > runtime_config block").
- the gradient-trap floors ``GRADIENT_NORM_FLOOR_BOUNDARY`` and
  ``GRADIENT_NORM_FLOOR_COILS``.
- the iota promotion surface default ``s = 0.24``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


# The VMEC_SINGLE_STAGE folder is not on ``sys.path`` for the simsopt-surrogate
# pytest run; load by file path. This matches the convention used by other
# tests that import optional example modules.
_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "VMEC_SINGLE_STAGE"
    / "vmec_single_stage_config.py"
)


def _load_config_module() -> ModuleType:
    module_name = "vmec_single_stage_config_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load vmec_single_stage_config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CONFIG_MODULE = _load_config_module()
VmecSingleStageConfig = _CONFIG_MODULE.VmecSingleStageConfig


# ---------------------------------------------------------------------------
# Frozen dataclass contract
# ---------------------------------------------------------------------------


def test_vmec_single_stage_config_is_dataclass() -> None:
    assert dataclasses.is_dataclass(VmecSingleStageConfig)


def test_vmec_single_stage_config_is_frozen() -> None:
    params = VmecSingleStageConfig.__dataclass_params__
    assert params.frozen is True


def test_vmec_single_stage_config_mutation_raises_frozen_instance_error() -> None:
    instance = VmecSingleStageConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        # Field is a typed primitive; assignment must be rejected on a
        # frozen dataclass.
        instance.w_coils = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bring-up placeholder defaults
# ---------------------------------------------------------------------------


def test_default_ns_array_matches_bring_up_schedule() -> None:
    instance = VmecSingleStageConfig()
    assert instance.ns_array == (13, 25, 51, 75, 101)


def test_default_iota_tol_matches_bring_up_placeholder() -> None:
    instance = VmecSingleStageConfig()
    assert instance.iota_tol == 0.02


def test_default_eps_fd_xsurface_abs_matches_plan_value() -> None:
    instance = VmecSingleStageConfig()
    assert instance.eps_fd_xsurface_abs == 1.0e-7


def test_default_eps_fd_xcoils_rel_matches_plan_value() -> None:
    instance = VmecSingleStageConfig()
    assert instance.eps_fd_xcoils_rel == 1.0e-4


# ---------------------------------------------------------------------------
# Gradient-trap floors
# ---------------------------------------------------------------------------


def test_gradient_norm_floor_boundary_constant() -> None:
    assert _CONFIG_MODULE.GRADIENT_NORM_FLOOR_BOUNDARY == 1.0e-8


def test_gradient_norm_floor_coils_constant() -> None:
    assert _CONFIG_MODULE.GRADIENT_NORM_FLOOR_COILS == 1.0e-8


# ---------------------------------------------------------------------------
# Iota promotion surface SSOT
# ---------------------------------------------------------------------------


def test_iota_promotion_surface_default_is_s_equals_0p24() -> None:
    instance = VmecSingleStageConfig()
    assert instance.iota_promotion_surface_s == 0.24
