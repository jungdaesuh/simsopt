"""Source-owned contract for the exact VMEC-host/JAX-device mirror."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_optimization.py"


def _module() -> ast.Module:
    return ast.parse(EXAMPLE.read_text(encoding="utf-8"))


def test_vmec_hybrid_has_the_exact_native_mirror_path() -> None:
    assert EXAMPLE.is_file()


def test_vmec_hybrid_keeps_vmec_on_host_and_jax_slice_explicit() -> None:
    module = _module()
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "Vmec" in imported_names
    assert "BiotSavartJAX" in imported_names
    assert "minimize_lbfgs_host_core" in imported_names
    assert "pure_callback" not in called_attributes


def test_vmec_hybrid_reports_separate_host_and_jax_slice_evidence() -> None:
    observable_keys = {
        key.value
        for node in ast.walk(_module())
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    required = {
        "execution_scope",
        "vmec_platform",
        "jax_platform",
        "vmec_elapsed_seconds",
        "jax_elapsed_seconds",
        "boundary_sha256",
        "vmec_objective",
        "stage_two_objective",
        "mixed_surface_gradient",
        "solution",
        "gradient",
    }
    assert required <= observable_keys
