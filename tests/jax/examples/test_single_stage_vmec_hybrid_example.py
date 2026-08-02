"""Source-owned contract for the exact VMEC-host/JAX-device mirror."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_optimization.py"
NATIVE_EXAMPLE = ROOT / "examples" / "3_Advanced" / "single_stage_optimization.py"
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


def test_vmec_hybrid_preserves_native_normalized_current_coordinates() -> None:
    current_expressions = [
        node.value.elt
        for node in ast.walk(_module())
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "base_currents"
        and isinstance(node.value, ast.ListComp)
    ]
    assert len(current_expressions) == 1
    expression = current_expressions[0]

    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.op, ast.Mult)
    assert isinstance(expression.left, ast.Call)
    assert isinstance(expression.left.func, ast.Name)
    assert expression.left.func.id == "Current"
    assert len(expression.left.args) == 1
    assert isinstance(expression.left.args[0], ast.Constant)
    assert expression.left.args[0].value == 1.0
    assert isinstance(expression.right, ast.Constant)
    assert expression.right.value == 1.0e5


def test_vmec_hybrid_preserves_native_identity_msc_penalty() -> None:
    matching_keywords = [
        keyword
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "StageTwoObjectiveConfig"
        for keyword in node.keywords
        if keyword.arg == "mean_squared_curvature_target_mode"
    ]

    assert len(matching_keywords) == 1
    value = matching_keywords[0].value
    assert isinstance(value, ast.Constant)
    assert value.value == "identity"


def test_vmec_hybrid_omits_native_unused_arclength_regularization() -> None:
    native_module = ast.parse(NATIVE_EXAMPLE.read_text(encoding="utf-8"))
    native_objective = next(
        node.value
        for node in native_module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "JF"
            for target in node.targets
        )
    )
    native_objective_names = {
        node.id for node in ast.walk(native_objective) if isinstance(node, ast.Name)
    }
    matching_keywords = [
        keyword
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "StageTwoObjectiveConfig"
        for keyword in node.keywords
        if keyword.arg == "arclength_variation_weight"
    ]

    assert "J_ALS" not in native_objective_names
    assert matching_keywords == []


def test_vmec_hybrid_fingerprint_binds_the_stage_two_contract() -> None:
    fingerprint_functions = [
        node
        for node in ast.walk(_module())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fingerprint"
    ]
    assert len(fingerprint_functions) == 1
    fingerprint_constants = {
        node.value
        for node in ast.walk(fingerprint_functions[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "stage_two" in fingerprint_constants
    assert "coil_preoptimization_solver" in fingerprint_constants

    fingerprint_calls = [
        node
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fingerprint"
    ]
    assert len(fingerprint_calls) == 1
    assert {keyword.arg for keyword in fingerprint_calls[0].keywords} >= {
        "coil_preoptimization_solver",
        "stage_two_config",
    }


def test_vmec_hybrid_parity_uses_scipy_trajectory_jax_lbfgsb() -> None:
    module = _module()
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    target_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "target_minimize"
    ]

    assert "get_backend_mode" in imported_names
    assert "target_minimize" in imported_names
    assert len(target_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in target_calls[0].keywords}
    method = keywords["method"]
    value_and_grad = keywords["value_and_grad"]
    assert isinstance(method, ast.Constant)
    assert method.value == "lbfgs-ondevice"
    assert isinstance(value_and_grad, ast.Constant)
    assert value_and_grad.value is True
    assert any(
        isinstance(node, ast.Constant) and node.value == "_parity"
        for node in ast.walk(module)
    )


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
    assert "validate_hybrid_jax_evaluation" in imported_names
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_hybrid_jax_evaluation" in called_names
    for callback_name in ("pure_callback", "io_callback"):
        assert callback_name not in imported_names
        assert callback_name not in called_attributes
        assert callback_name not in called_names


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
        "coil_preoptimization_evaluations",
        "coil_preoptimization_final_gradient",
        "coil_preoptimization_final_objective",
        "coil_preoptimization_initial_gradient",
        "coil_preoptimization_initial_objective",
        "coil_preoptimization_iterations",
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
