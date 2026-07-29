"""Source-owned contract for the exact VMEC-host/JAX-device mirror."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_optimization.py"
AUTHORITY_WORKFLOW = ROOT / ".github" / "workflows" / "jax_vmec_hybrid_authority.yml"


def _module() -> ast.Module:
    return ast.parse(EXAMPLE.read_text(encoding="utf-8"))


def test_vmec_hybrid_has_the_exact_native_mirror_path() -> None:
    assert EXAMPLE.is_file()
    assignments = {
        target.id: node.value.value
        for node in _module().body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert assignments["EXAMPLE_ID"] == "native-single-stage-optimization"


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
    assert "scalar_example_driver" in imported_names
    assert "minimize_bfgs_host_core" in imported_names
    assert "minimize_lbfgs_host_core" in imported_names
    assert "line_search_value_and_grad_more_thuente_host" in imported_names
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


def test_vmec_hybrid_authority_is_manual_and_build_identity_bound() -> None:
    workflow = AUTHORITY_WORKFLOW.read_text(encoding="utf-8")

    trigger = workflow.split("jobs:", maxsplit=1)[0]
    assert "workflow_dispatch:" in trigger
    assert "schedule:" not in trigger
    assert "pull_request:" not in trigger
    assert "push:" not in trigger
    assert "expected_vmec_sha256:" in trigger
    assert "expected_mpi_world_size:" in trigger
    assert "VMEC Python extension SHA-256 mismatch" in workflow
    assert "configuration_sha256" in workflow
    assert "execution_scope" in workflow
    assert "jax_slice_gpu" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "--smoke" in workflow
    assert "--device cpu" not in workflow
    assert "--device gpu" not in workflow
    assert "--intent parity" not in workflow
    assert "--scale bounded" not in workflow
