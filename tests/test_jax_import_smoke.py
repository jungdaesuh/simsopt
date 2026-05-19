"""
Import smoke tests for the JAX code path.

These tests verify that JAX modules can be imported through the real
``simsopt`` package entrypoints (not via ``importlib.util`` bypass).
They run in the no-simsoptpp environment to catch import-chain regressions.

Each test launches a fresh Python subprocess so that ``sys.modules`` is
guaranteed clean — other test modules in this repo inject package stubs
at import time, which would contaminate in-process imports.

This file also keeps a small number of process-isolated JAX runtime
regressions whose contract depends on a fresh subprocess. The historical
name stays for continuity, but larger functional subprocess programs
should live in real Python modules rather than inline ``python -c`` blobs.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pytest

# Resolve the src/ directory relative to the repo root so subprocesses
# can import simsopt without a pip install.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_OPTIMIZER_JAX_PATH = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax.py"
_OPTIMIZER_PRIVATE_DIR = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax_private"
_RUNTIME_BACKEND_PATH = Path(_SRC_DIR) / "simsopt" / "backend" / "runtime.py"
_CPU_RUN_CODE_BENCHMARK_PATH = (
    Path(_REPO_ROOT) / "benchmarks" / "cpu_run_code_benchmark.py"
)
_JAX_SUBPROCESS_CASES_PATH = (
    Path(_REPO_ROOT) / "tests" / "subprocess" / "jax_runtime_cases.py"
)
_IMPORT_SMOKE_CASES_PATH = (
    Path(_REPO_ROOT) / "tests" / "subprocess" / "import_smoke_cases.py"
)
_SINGLE_STAGE_SURFACE_REPROJECTION_PROBE_PATH = (
    Path(_REPO_ROOT) / "benchmarks" / "single_stage_surface_reprojection_probe.py"
)
_ONDEVICE_COLD_SMOKE_TIMEOUT = 300
_SINGLE_STAGE_TARGET_RUNTIME_COLD_SMOKE_TIMEOUT = 480
_ENTRYPOINT_RUNTIME_AUDIT_PATHS = (
    Path(_REPO_ROOT) / "benchmarks" / "biot_savart_kernel_scaling.py",
    Path(_REPO_ROOT) / "benchmarks" / "cpu_run_code_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "gpu_run_code_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "jax_derivative_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "jax_feasibility_spike.py",
    Path(_REPO_ROOT) / "benchmarks" / "optimistix_eval.py",
    (
        Path(_REPO_ROOT)
        / "examples"
        / "single_stage_optimization"
        / "SINGLE_STAGE"
        / "single_stage_banana_example.py"
    ),
    (
        Path(_REPO_ROOT)
        / "examples"
        / "single_stage_optimization"
        / "STAGE_2"
        / "banana_coil_solver.py"
    ),
)
_BACKEND_SELECTOR_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_DEBUG",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_DISABLE_JIT",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_GPU_PREALLOCATE",
    "SIMSOPT_JAX_GPU_MEM_FRACTION",
    "SIMSOPT_JAX_GPU_ALLOCATOR",
    "SIMSOPT_TF_GPU_ALLOCATOR",
    "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU",
    "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "SIMSOPT_JAX_SHARDING",
    "SIMSOPT_JAX_SHARDING_AXIS",
    "SIMSOPT_JAX_COIL_SHARDING_AXIS",
    "SIMSOPT_JAX_MIN_POINTS_TO_SHARD",
    "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD",
    "SIMSOPT_JAX_MIN_COILS_TO_SHARD",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_CLIENT_MEM_FRACTION",
    "TF_GPU_ALLOCATOR",
)

LegacyCurveObjectiveValueCase = Literal[
    "curve-length",
    "lp-curve-curvature",
    "curve-curve-distance",
    "curve-surface-distance",
    "lp-curve-curvature-barrier",
    "lp-curve-torsion",
    "framed-curve-twist",
]
LegacyCurveObjectiveGradientCase = Literal[
    "lp-curve-curvature-barrier",
    "lp-curve-curvature",
    "curve-curve-distance",
    "curve-surface-distance",
    "lp-curve-torsion",
    "framed-curve-twist",
]


def _build_clean_subprocess_env(
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in _BACKEND_SELECTOR_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env is not None:
        env.update(extra_env)
    return env


def _run_python_script(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a repo-local Python script in a clean subprocess."""
    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
        env=_build_clean_subprocess_env(extra_env),
    )
    return result.returncode, result.stderr.strip()


def _run_python_script_capture(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a repo-local Python script and return the completed process."""
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
        env=_build_clean_subprocess_env(extra_env),
    )


def _last_json_payload_or_none(stdout: str) -> dict[str, object] | None:
    """Return the last non-empty JSON object on subprocess stdout, if any."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None
    return None


def _maybe_skip_from_subprocess_stdout(stdout: str, *, expected_case: str) -> None:
    """Honor a ``{"skipped": true, ...}`` sentinel by raising ``pytest.skip``."""
    payload = _last_json_payload_or_none(stdout)
    if payload is None:
        return
    if payload.get("skipped") is not True:
        return
    assert payload.get("checked") is False, (
        f"{expected_case}: skipped sentinel must have checked=False, got {payload!r}"
    )
    assert payload.get("case") == expected_case, (
        f"{expected_case}: skipped sentinel case mismatch, got {payload!r}"
    )
    skip_reason = payload.get("skip_reason")
    assert isinstance(skip_reason, str) and skip_reason, (
        f"{expected_case}: skipped sentinel must carry a non-empty skip_reason, got {payload!r}"
    )
    pytest.skip(f"{expected_case}: {skip_reason}")


def _run_python_script_json_payload(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    failure_message: str,
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
    expected_case: str | None = None,
) -> dict[str, object]:
    """Run a subprocess smoke case and parse the final JSON stdout line.

    If ``expected_case`` is provided and the subprocess emits a skip sentinel,
    this raises ``pytest.skip`` instead of returning the sentinel payload.
    """
    result = _run_python_script_capture(
        script_path,
        args=args,
        timeout=timeout,
        extra_env=extra_env,
    )
    assert result.returncode == 0, f"{failure_message}:\n{result.stderr.strip()}"
    if expected_case is not None:
        _maybe_skip_from_subprocess_stdout(
            result.stdout,
            expected_case=expected_case,
        )
    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert stdout_lines, f"{failure_message}: subprocess emitted no JSON payload"
    payload = json.loads(stdout_lines[-1])
    assert isinstance(payload, dict), f"{failure_message}: payload is not an object"
    return payload


def _assert_subprocess_json_sentinel(
    payload: dict[str, object],
    *,
    expected_case: str,
    failure_message: str,
) -> None:
    assert payload["case"] == expected_case, failure_message
    if payload["skipped"] is True:
        assert payload["checked"] is False, failure_message
        skip_reason = payload["skip_reason"]
        assert isinstance(skip_reason, str) and skip_reason, failure_message
        pytest.skip(f"{expected_case}: {skip_reason}")
    assert payload["skipped"] is False, failure_message
    assert payload["checked"] is True, failure_message
    assert isinstance(payload["invariant"], str) and payload["invariant"], (
        failure_message
    )
    assert isinstance(payload["loaded_module_count"], int), failure_message
    assert isinstance(payload["simsopt_module_count"], int), failure_message


def _assert_import_smoke_case_passes(case_name: str, failure_message: str) -> None:
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(case_name,),
        failure_message=failure_message,
    )


def _assert_python_script_passes(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    failure_message: str,
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> None:
    if script_path == _IMPORT_SMOKE_CASES_PATH:
        expected_case = str(args[0])
        payload = _run_python_script_json_payload(
            script_path,
            args=args,
            failure_message=failure_message,
            timeout=timeout,
            extra_env=extra_env,
        )
        _assert_subprocess_json_sentinel(
            payload,
            expected_case=expected_case,
            failure_message=failure_message,
        )
        return
    result = _run_python_script_capture(
        script_path,
        args=args,
        timeout=timeout,
        extra_env=extra_env,
    )
    assert result.returncode == 0, f"{failure_message}:\n{result.stderr.strip()}"
    if script_path == _JAX_SUBPROCESS_CASES_PATH:
        _maybe_skip_from_subprocess_stdout(
            result.stdout,
            expected_case=str(args[0]),
        )


def _find_private_jax_src_usages(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    jax_names = {"jax"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax":
                    jax_names.add(alias.asname or alias.name)
    usages: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            usages.extend(
                f"{alias.name} @ L{node.lineno}"
                for alias in node.names
                if alias.name.startswith("jax._src")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("jax._src"):
                usages.append(f"{module} @ L{node.lineno}")
            elif module == "jax":
                usages.extend(
                    f"from jax import {alias.name} @ L{node.lineno}"
                    for alias in node.names
                    if alias.name == "_src"
                )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "_src"
            and isinstance(node.value, ast.Name)
            and node.value.id in jax_names
        ):
            usages.append(f"{node.value.id}._src @ L{node.lineno}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_src"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in jax_names
        ):
            usages.append(f'getattr({node.args[0].id}, "_src") @ L{node.lineno}')
    return usages


def _assert_no_private_jax_src_usage(path: Path, *, label: str) -> None:
    forbidden_usages = _find_private_jax_src_usages(path)
    assert forbidden_usages == [], f"{label} must not use jax._src: {forbidden_usages}"


def _find_import_line(path: Path, module_name: str) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    import_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                import_lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                import_lines.append(node.lineno)
    return min(import_lines) if import_lines else None


def _find_named_call_lines(path: Path, function_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    call_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == function_name:
                call_lines.append(node.lineno)
    return sorted(call_lines)


def test_find_private_jax_src_usages_detects_alias_attribute_access(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        'import jax as jj\nvalue = jj._src\nshadow = getattr(jj, "_src")\n',
        encoding="utf-8",
    )

    usages = _find_private_jax_src_usages(path)

    assert "jj._src @ L2" in usages
    assert 'getattr(jj, "_src") @ L3' in usages


def test_import_package_root():
    """simsopt package imports without simsoptpp."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_import_package_root",),
        failure_message="import simsopt failed",
    )


def test_import_package_root_without_generated_version_file():
    """Raw source imports should tolerate a missing generated _version.py."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_import_package_root_without_generated_version_file",),
        failure_message="raw source import should not require generated _version.py",
    )


def test_repo_bootstrap_synthesizes_version_for_clean_source_tree():
    """repo_bootstrap should tolerate source trees without generated _version.py."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_synthesizes_version_for_clean_source_tree",),
        failure_message="repo_bootstrap clean-source version smoke failed",
    )


def test_repo_bootstrap_is_idempotent_for_local_source_tree():
    """Repeated bootstrap calls must not churn class identity for local imports."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_is_idempotent_for_local_source_tree",),
        failure_message="repo_bootstrap should be idempotent for local source imports",
    )


def test_root_conftest_imports_without_jax_installed():
    """Root test fixtures must not fail collection in non-JAX environments."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_root_conftest_imports_without_jax_installed",),
        failure_message="root tests/conftest.py should import cleanly without JAX",
    )


def test_root_conftest_bootstraps_local_simsopt_over_foreign_resolution():
    """Root fixtures must pin imports to this repo even when another simsopt is earlier."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_root_conftest_bootstraps_local_simsopt_over_foreign_resolution",),
        failure_message=(
            "root tests/conftest.py should bootstrap the local simsopt package"
        ),
    )


def test_repo_bootstrap_purges_detached_local_submodules():
    """A second bootstrap must purge detached ``simsopt.*`` submodules."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_purges_detached_local_submodules",),
        failure_message="repo_bootstrap should purge detached local submodules",
    )


def test_repo_bootstrap_strips_editable_meta_path_finders_on_fast_path():
    """Warm bootstraps must remove editable finders before later submodule imports."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_strips_editable_meta_path_finders_on_fast_path",),
        failure_message="repo_bootstrap should strip editable meta_path finders",
    )


def test_repo_bootstrap_preserves_unrelated_editable_meta_path_finders():
    """Warm bootstraps must not remove editable finders for unrelated packages."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_preserves_unrelated_editable_meta_path_finders",),
        failure_message="repo_bootstrap should preserve unrelated editable finders",
    )


def test_repo_bootstrap_reloads_local_simsoptpp_over_foreign_module():
    """Bootstrapping local simsopt must replace foreign ``simsoptpp`` modules."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_repo_bootstrap_reloads_local_simsoptpp_over_foreign_module",),
        failure_message="repo_bootstrap should replace foreign simsoptpp modules",
    )


def test_import_package_root_native_cpu_does_not_require_jax_runtime():
    """Importing package root without JAX selectors must not force a JAX import."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_import_package_root_native_cpu_does_not_require_jax_runtime",),
        failure_message="package root import unexpectedly required jax",
    )


def test_package_root_jax_selector_propagates_missing_jax():
    """Explicit JAX backend selection must fail loudly when JAX cannot import."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_package_root_jax_selector_propagates_missing_jax",),
        failure_message="explicit JAX selector masked missing jax",
    )


def test_package_root_propagates_backend_import_error():
    """Package-root import must not mask internal backend ImportErrors."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_package_root_propagates_backend_import_error",),
        failure_message="package root masked backend import failure",
    )


def test_entrypoint_runtime_helper_configures_cpu_before_import():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_entrypoint_runtime_helper_configures_cpu_before_import",),
        failure_message="entrypoint runtime helper should pin CPU before importing jax",
    )


def test_entrypoint_runtime_helper_auto_clears_stale_platform_env():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_entrypoint_runtime_helper_auto_clears_stale_platform_env",),
        failure_message="entrypoint runtime helper should clear stale platform env when auto is requested",
    )


def test_entrypoint_runtime_helper_adds_detected_cuda_toolchain_root():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_entrypoint_runtime_helper_adds_detected_cuda_toolchain_root",),
        failure_message="entrypoint runtime helper should auto-detect a CUDA toolchain root",
    )


def test_entrypoint_runtime_helper_accepts_multi_platform_env_list():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_entrypoint_runtime_helper_accepts_multi_platform_env_list",),
        failure_message="entrypoint runtime helper should preserve multi-platform JAX_PLATFORMS lists",
    )


def test_entrypoint_runtime_helper_promotes_cuda_to_cuda_cpu_for_callback_flags():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_entrypoint_runtime_helper_promotes_cuda_to_cuda_cpu_for_callback_flags",
        ),
        failure_message="entrypoint runtime helper should keep a CPU lane for explicit diagnostic callback runs",
    )


def test_run_code_benchmark_common_import_is_jax_cold():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_run_code_benchmark_common_import_is_jax_cold",),
        failure_message="run_code_benchmark_common import should not initialize jax",
    )


def test_cpu_run_code_benchmark_pins_cpu_before_import():
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_cpu_run_code_benchmark_pins_cpu_before_import",),
        failure_message="cpu_run_code_benchmark should request CPU before importing jax",
    )


def test_audited_entrypoints_configure_runtime_before_importing_jax():
    for path in _ENTRYPOINT_RUNTIME_AUDIT_PATHS:
        configure_lines = _find_named_call_lines(
            path, "configure_entrypoint_jax_runtime"
        )
        first_jax_import = _find_import_line(path, "jax")

        assert configure_lines, (
            f"{path.name} must call configure_entrypoint_jax_runtime"
        )
        assert first_jax_import is not None, f"{path.name} must import jax explicitly"
        assert min(configure_lines) < first_jax_import, (
            f"{path.name} must configure the JAX runtime before importing jax"
        )


def test_programmatic_backend_selection_configures_jax_runtime():
    """The public config API should support the new mode-based backend contract."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_programmatic_backend_selection_configures_jax_runtime",),
        failure_message="programmatic backend config failed",
    )


def test_programmatic_backend_persistent_cache_writes_small_kernel():
    """Runtime cache thresholds should let even a tiny compiled kernel persist."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_programmatic_backend_persistent_cache_writes_small_kernel",),
        failure_message="persistent cache write smoke failed",
    )


def test_parity_mode_defaults_transfer_guard_and_keeps_x64_enabled():
    """Parity modes should own x64 and transfer-guard defaults without extra flags."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_parity_mode_defaults_transfer_guard_and_keeps_x64_enabled",),
        failure_message="parity mode guardrail contract failed",
    )


def test_env_selected_guardrails_eagerly_configure_jax_runtime():
    """Import-time eager config should honor parity x64/debug-nans/transfer-guard envs."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_env_selected_guardrails_eagerly_configure_jax_runtime",),
        failure_message="eager guardrail config failed",
    )


def test_transfer_guard_disallow_rejects_implicit_host_to_device_jit_inputs():
    """Disallow mode should catch implicit NumPy->JAX transfers at a JIT boundary."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_rejects_implicit_host_to_device_jit_inputs",
        ),
        failure_message="transfer-guard disallow smoke failed",
    )


def test_transfer_guard_disallow_allows_target_backend_x64_guard():
    """Target-lane x64 checks must not allocate JAX arrays under disallow mode."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_target_backend_x64_guard",),
        failure_message="target-backend x64 guard should be transfer-clean",
    )


def test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes():
    """Private ondevice L-BFGS lanes must stay transfer-clean under disallow."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes",),
        failure_message="lbfgs-ondevice transfer-guard smoke failed",
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
    )


def test_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry():
    """Direct target_minimize() should stay transfer-clean for structured pytrees."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry",
        ),
        failure_message="target_minimize structured pytree disallow smoke failed",
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
    )


def test_transfer_guard_disallow_allows_surface_surface_distance_smoke():
    """SurfaceSurfaceDistance must place host gamma arrays explicitly under disallow."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_surface_surface_distance_smoke",),
        failure_message="SurfaceSurfaceDistance transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_enforces_single_stage_target_runtime_boundaries():
    """Single-stage runtime boundaries must allow only the explicit staging seam."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("single-stage-target-runtime-transfer-guard",),
        failure_message=("single-stage target runtime transfer-guard contract drifted"),
        timeout=_SINGLE_STAGE_TARGET_RUNTIME_COLD_SMOKE_TIMEOUT,
    )


def _assert_ondevice_optimizer_reuses_compiled_solver(method: str) -> None:
    payload = _run_python_script_json_payload(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("compile-count", method),
        failure_message=f"{method} compile-count smoke failed",
        timeout=150 if method == "lbfgs-ondevice" else 30,
        extra_env={"JAX_ENABLE_COMPILATION_CACHE": "0"},
        expected_case="compile-count",
    )
    assert payload == {
        "case": "compile-count",
        "method": method,
        "compile_count": 0 if method == "lbfgs-ondevice" else 1,
        "run_count": 3,
    }


def test_lbfgs_ondevice_reuses_compiled_solver_across_identical_calls():
    """Repeated lbfgs-ondevice calls must not compile solver control kernels."""
    _assert_ondevice_optimizer_reuses_compiled_solver("lbfgs-ondevice")


def test_bfgs_ondevice_reuses_compiled_solver_across_identical_calls():
    """Repeated identical bfgs-ondevice calls must not recompile run_solver."""
    _assert_ondevice_optimizer_reuses_compiled_solver("bfgs-ondevice")


def test_target_lbfgs_ondevice_reuses_compiled_solver_across_identical_value_and_grad_calls():
    """Target-lane lbfgs-ondevice must not compile solver control kernels."""
    payload = _run_python_script_json_payload(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("target-compile-count",),
        failure_message=(
            "target lbfgs-ondevice value-and-grad compile-count smoke failed"
        ),
        timeout=150,
        extra_env={"JAX_ENABLE_COMPILATION_CACHE": "0"},
        expected_case="target-compile-count",
    )
    assert payload == {
        "case": "target-compile-count",
        "method": "lbfgs-ondevice",
        "compile_count": 0,
        "run_count": 3,
        "value_and_grad": True,
    }


def test_stage2_target_outer_loop_reuses_compiled_solver_across_identical_calls():
    """Real Stage 2 target-lane outer-loop must keep solver control on host."""
    payload = _run_python_script_json_payload(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("stage2-target-compile-count",),
        failure_message="Stage 2 target outer-loop compile-count smoke failed",
        extra_env={
            "JAX_ENABLE_COMPILATION_CACHE": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
        expected_case="stage2-target-compile-count",
    )
    assert payload == {
        "case": "stage2-target-compile-count",
        "method": "lbfgs-ondevice",
        "compile_count": 0,
        "run_count": 3,
        "value_and_grad": True,
    }


def test_ondevice_solver_cache_respects_mutable_objective_state():
    """Unmarked mutable callables must retrace so updated host state is observed."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("mutable-objective-state",),
        failure_message="ondevice solver cache must not freeze mutable objective state",
    )


def test_structured_ondevice_solver_cache_respects_mutable_objective_state():
    """Structured pytree entry must not freeze mutable cacheable objective state."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("structured-mutable-objective-state",),
        failure_message=(
            "structured ondevice solver cache must not freeze mutable objective state"
        ),
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
    )


def test_transfer_guard_disallow_allows_adam_ondevice_quadratic_smokes():
    """Public ondevice Adam lane must stay transfer-clean under disallow."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_adam_ondevice_quadratic_smokes",),
        failure_message="adam-ondevice transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_lm_ondevice_quadratic_smokes():
    """Ondevice LM least-squares must stay transfer-clean under disallow."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_lm_ondevice_quadratic_smokes",),
        failure_message="lm-ondevice transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_target_least_squares_structured_entry():
    """Direct target_least_squares() should stay transfer-clean for structured pytrees."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_allows_target_least_squares_structured_entry",
        ),
        failure_message="target_least_squares structured pytree disallow smoke failed",
    )


def test_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants():
    """Ondevice optimizer loops must compile even when objectives capture host arrays."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants",
        ),
        failure_message="ondevice optimizer loop closure-constant smoke failed",
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
    )


def test_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants():
    """GPU ondevice optimizers must not capture device-backed compile constants."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants",
        ),
        failure_message="GPU ondevice optimizer transfer-guard smoke failed",
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_traceable_newton_with_host_closure_constants():
    """Traceable Newton helpers must not eagerly cross host/device boundaries."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_allows_traceable_newton_with_host_closure_constants",
        ),
        failure_message="traceable Newton closure-constant smoke failed",
    )


def test_transfer_guard_disallow_allows_boozer_residual_host_scalars():
    """Boozer residual kernels must explicitly materialize legacy host scalars."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_boozer_residual_host_scalars",),
        failure_message="boozer residual host-scalar transfer smoke failed",
    )


def test_transfer_guard_disallow_allows_boozer_decision_vector_split():
    """Boozer decision-vector split must not create implicit index transfers."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_boozer_decision_vector_split",),
        failure_message="boozer decision-vector split transfer smoke failed",
    )


def test_transfer_guard_disallow_allows_biot_savart_point_chunking():
    """Point-chunked Biot-Savart B/A kernels must stay traceable under JAX loops."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("biot-savart-point-chunking",),
        failure_message="Biot-Savart point-chunking smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_spec_eval():
    """GPU grouped-field kernels must not close over device-backed selector constants."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-gpu-spec-eval",),
        failure_message="grouped Biot-Savart GPU spec transfer-guard smoke failed",
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_grouped_biot_savart_accepts_explicit_point_sharding():
    """Grouped-field kernels should accept explicitly sharded point clouds."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-explicit-point-sharding",),
        failure_message="grouped Biot-Savart explicit point sharding smoke failed",
    )


def test_grouped_biot_savart_coil_collective_parity_and_lowering():
    """Coil-axis grouped-field collectives must lower to an all-reduce."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-coil-collective",),
        failure_message="grouped Biot-Savart coil collective smoke failed",
        timeout=120,
        extra_env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
            "SIMSOPT_JAX_SHARDING": "coil_groups",
            "SIMSOPT_JAX_MIN_COILS_TO_SHARD": "1",
            "SIMSOPT_JAX_TRANSFER_GUARD": "allow",
        },
    )


def test_pairwise_penalty_accepts_explicit_row_sharding():
    """Pairwise penalty kernels should accept explicitly sharded row-owned inputs."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("pairwise-penalty-explicit-row-sharding",),
        failure_message="pairwise penalty explicit row sharding smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_current_arrays():
    """Grouped coil specs should accept staged current arrays without Python indexing."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-gpu-current-arrays",),
        failure_message="grouped Biot-Savart GPU current-array transfer-guard smoke failed",
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_host_scalar_currents():
    """Grouped coil specs should explicitly stage host scalar currents on GPU."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-host-scalar-currents",),
        failure_message="grouped Biot-Savart host-scalar current transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_host_spec_vjp():
    """Host-backed grouped coil specs must remain usable in eager VJP paths."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("grouped-host-spec-vjp",),
        failure_message="grouped Biot-Savart host-spec VJP transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_preserves_shifted_grid_axis_sample():
    """Shifted quadrature grids must use the sampled surface point for axis-z."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("shifted-grid-axis-sample",),
        failure_message="shifted-grid axis sample smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_init():
    """CurveCWSFourierCPP should explicitly materialize quadpoints under disallow mode."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("curvecwsfouriercpp-init",),
        failure_message="CurveCWSFourierCPP transfer-guard init smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_curve_length_gradient():
    """CurveCWSFourierCPP length gradient should use explicit host/device boundaries."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("curvecwsfouriercpp-curve-length-gradient",),
        failure_message="CurveCWSFourierCPP CurveLength transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_curveperturbed_init():
    """CurvePerturbed should explicitly place sampled host arrays under disallow."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("curveperturbed-init",),
        failure_message="CurvePerturbed transfer-guard init smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_curve_distance_gradient():
    """CurveCWSFourierCPP distance gradients should materialize JAX geometry before slicing."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("curvecwsfouriercpp-curve-distance-gradient",),
        failure_message="CurveCWSFourierCPP CurveCurveDistance transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_stage2_target_objective_host_closure_constants():
    """Direct Stage 2 objective evaluation must tolerate strict transfer guard."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("stage2-target-objective-host-closure-constants",),
        failure_message="Stage 2 direct objective transfer-guard smoke failed",
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_stage2_target_objective_ondevice_entry():
    """The real ondevice optimizer entry must tolerate strict transfer guard."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("stage2-target-objective-ondevice-entry",),
        failure_message="Stage 2 ondevice transfer-guard entry smoke failed",
        timeout=_ONDEVICE_COLD_SMOKE_TIMEOUT,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_gamma_2d_eager_host_constants():
    """Eager curve geometry helpers must keep host literals explicit under strict guard."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("gamma-2d-eager-host-constants",),
        failure_message="gamma_2d strict transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_closed_curve_self_intersection_summary():
    """Strict GPU geometry probes must not materialize shape scalars on the host."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("closed-curve-self-intersection-summary",),
        failure_message="closed-curve self-intersection strict transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_single_stage_surface_self_intersection():
    """Single-stage supported-surface self-intersection should stay transfer-clean."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("single-stage-surface-self-intersection",),
        failure_message="single-stage surface self-intersection transfer-guard smoke failed",
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_segment_segment_distance_pure_rejects_host_numpy_inputs_without_spec():
    from simsopt.jax_core.curve_geometry import segment_segment_distance_pure

    point = np.zeros(3, dtype=np.float64)
    with pytest.raises(TypeError, match="JAX/spec-backed arrays"):
        segment_segment_distance_pure(point, point, point, point)


def test_transfer_guard_disallow_allows_surface_xyztensorfourier_gamma_from_dofs():
    """SurfaceXYZTensorFourier geometry should stay clean in eager and jitted strict-guard lanes."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("surface-xyztensorfourier-gamma-from-dofs",),
        failure_message="SurfaceXYZTensorFourier gamma strict transfer-guard smoke failed",
        timeout=120,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_project_surface_dofs_to_resolution():
    """Warm-start surface reprojection should keep host/device transfers explicit."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("project-surface-dofs-to-resolution",),
        failure_message="project_surface_dofs_to_resolution strict transfer-guard smoke failed",
    )


def test_single_stage_surface_reprojection_probe_emits_structured_cpu_result(tmp_path):
    """The staged reprojection probe should complete on CPU and write stage metadata."""
    output_json = tmp_path / "single_stage_surface_reprojection_probe.json"

    rc, err = _run_python_script(
        _SINGLE_STAGE_SURFACE_REPROJECTION_PROBE_PATH,
        args=("--platform", "cpu", "--output-json", str(output_json)),
        timeout=180,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )

    assert rc == 0, f"single-stage reprojection probe failed:\n{err}"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failure_stage"] is None
    assert [stage["name"] for stage in payload["stages"]] == [
        "load_source_surface",
        "device_put_source_dofs",
        "surface_rz_fourier_spec_from_dofs",
        "surface_rz_fourier_gamma_from_spec",
        "surface_rz_fourier_gamma_from_dofs",
        "project_surface_dofs_to_resolution",
    ]


def test_transfer_guard_disallow_allows_coil_symmetry_spec_identity_default():
    """Coil symmetry defaults should build the identity rotation explicitly."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("coil-symmetry-spec-identity-default",),
        failure_message="coil symmetry identity default should be transfer-clean",
    )


_LEGACY_CURVE_OBJECTIVE_VALUE_CASES: tuple[
    tuple[str, LegacyCurveObjectiveValueCase],
    ...,
] = (
    ("CurveLength", "curve-length"),
    ("LpCurveCurvature", "lp-curve-curvature"),
    ("CurveCurveDistance", "curve-curve-distance"),
    ("CurveSurfaceDistance", "curve-surface-distance"),
    ("LpCurveCurvatureBarrier", "lp-curve-curvature-barrier"),
    ("LpCurveTorsion", "lp-curve-torsion"),
    ("FramedCurveTwist", "framed-curve-twist"),
)

_LEGACY_CURVE_OBJECTIVE_GRADIENT_CASES: tuple[
    tuple[str, LegacyCurveObjectiveGradientCase],
    ...,
] = (
    ("LpCurveCurvatureBarrier", "lp-curve-curvature-barrier"),
    ("LpCurveCurvature", "lp-curve-curvature"),
    ("CurveCurveDistance", "curve-curve-distance"),
    ("CurveSurfaceDistance", "curve-surface-distance"),
    ("LpCurveTorsion", "lp-curve-torsion"),
    ("FramedCurveTwist", "framed-curve-twist"),
)


@pytest.mark.parametrize(
    ("label", "objective_case"),
    _LEGACY_CURVE_OBJECTIVE_VALUE_CASES,
)
def test_transfer_guard_disallow_allows_legacy_curve_objective_values(
    label: str,
    objective_case: LegacyCurveObjectiveValueCase,
):
    """Legacy curve objectives must use explicit host/device boundaries under disallow."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("legacy-curve-objective-value", objective_case),
        failure_message=f"{label} transfer-guard value smoke failed",
    )


@pytest.mark.parametrize(
    ("label", "objective_case"),
    _LEGACY_CURVE_OBJECTIVE_GRADIENT_CASES,
)
def test_transfer_guard_disallow_allows_legacy_curve_objective_gradients(
    label: str,
    objective_case: LegacyCurveObjectiveGradientCase,
):
    """Legacy curve-objective gradients must keep host/device transfers explicit."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("legacy-curve-objective-gradient", objective_case),
        failure_message=f"{label} transfer-guard gradient smoke failed",
    )


def test_transfer_guard_disallow_allows_pairwise_curve_penalty_pure_functions():
    """Pure pairwise penalty helpers must not materialize host scalars implicitly."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("pairwise-curve-penalty-pure-functions",),
        failure_message=(
            "pairwise curve penalty pure functions should stay transfer-clean"
        ),
    )


def test_transfer_guard_disallow_allows_surfacerzfourier_spec_defaults():
    """SurfaceRZFourier spec defaults should avoid zeros_like scalar materialization."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("surfacerzfourier-spec-defaults",),
        failure_message="SurfaceRZFourier transfer-guard spec smoke failed",
    )


def test_transfer_guard_disallow_allows_surface_rzfourier_gamma_from_spec():
    """Surface gamma evaluation should avoid implicit eager scalar transfers."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("surface-rzfourier-gamma-from-spec",),
        failure_message="SurfaceRZFourier gamma transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_surface_rzfourier_normal_from_spec():
    """Surface normal evaluation should stay transfer-clean under disallow mode."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("surface-rzfourier-normal-from-spec",),
        failure_message="SurfaceRZFourier normal transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_squaredfluxjax_construction():
    """SquaredFluxJAX construction should not fail in fixed-surface setup."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_squaredfluxjax_construction",),
        failure_message="SquaredFluxJAX transfer-guard construction smoke failed",
    )


def test_transfer_guard_disallow_rejects_squaredfluxjax_surface_without_spec():
    """SquaredFluxJAX must require immutable surface specs in strict parity lanes."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=(
            "case_transfer_guard_disallow_rejects_squaredfluxjax_surface_without_spec",
        ),
        failure_message="SquaredFluxJAX missing-surface-spec rejection smoke failed",
    )


def test_transfer_guard_disallow_allows_clamped_xyztensor_surface_spec():
    """Clamped tensor surfaces should route through their immutable JAX spec."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_clamped_xyztensor_surface_spec",),
        failure_message="Clamped SurfaceXYZTensorFourier transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_lpcurveforce_shared_state_packing():
    """LpCurveForce shared-state packing must explicitly place host geometry on JAX arrays."""
    _assert_python_script_passes(
        _IMPORT_SMOKE_CASES_PATH,
        args=("case_transfer_guard_disallow_allows_lpcurveforce_shared_state_packing",),
        failure_message="LpCurveForce shared-state transfer-guard smoke failed",
    )


def test_native_cpu_backend_selection_does_not_require_jax_runtime():
    """native_cpu config must not force a JAX import when only CPU mode is selected."""
    _assert_import_smoke_case_passes(
        "case_native_cpu_backend_selection_does_not_require_jax_runtime",
        "native_cpu config unexpectedly required jax",
    )


def test_native_cpu_policy_does_not_configure_jax_without_selector():
    """Default/native policy does not mutate JAX x64 without a selector."""
    _assert_import_smoke_case_passes(
        "case_native_cpu_policy_does_not_configure_jax_without_selector",
        "native_cpu import unexpectedly configured JAX x64",
    )


def test_import_biotsavart_jax():
    """BiotSavartJAX is importable through the real package entrypoint."""
    _assert_import_smoke_case_passes(
        "case_import_biotsavart_jax",
        "import BiotSavartJAX failed",
    )


def test_import_jax_core_specs():
    """The pure JAX kernel-layer package imports through the real package tree."""
    _assert_import_smoke_case_passes(
        "case_import_jax_core_specs",
        "import simsopt.jax_core failed",
    )


def test_jax_core_specs_are_pytrees():
    """Immutable JAX specs must flatten and survive JIT as real pytrees."""
    _assert_import_smoke_case_passes(
        "case_jax_core_specs_are_pytrees",
        "jax_core pytree contract failed",
    )


def test_jax_core_grouped_field_chunking_matches_dense_sum():
    """Chunked grouped-field evaluation must preserve dense grouped parity."""
    _assert_import_smoke_case_passes(
        "case_jax_core_grouped_field_chunking_matches_dense_sum",
        "jax_core grouped chunking contract failed",
    )


def test_import_squaredflux_jax():
    """SquaredFluxJAX is importable through the real package entrypoint."""
    _assert_import_smoke_case_passes(
        "case_import_squaredflux_jax",
        "import SquaredFluxJAX failed",
    )


def test_import_boozersurface_jax():
    """BoozerSurfaceJAX is importable through the real package entrypoint."""
    _assert_import_smoke_case_passes(
        "case_import_boozersurface_jax",
        "import BoozerSurfaceJAX failed",
    )


def test_import_core_optimizable():
    """Optimizable base class imports without simsoptpp."""
    _assert_import_smoke_case_passes(
        "case_import_core_optimizable",
        "import Optimizable failed",
    )


def test_optimizer_jax_import_is_lazy():
    """Importing the public optimizer module must not eagerly load the private package."""
    _assert_import_smoke_case_passes(
        "case_optimizer_jax_import_is_lazy",
        "optimizer_jax lazy import check failed",
    )


def test_optimizer_jax_public_reference_methods_work_without_private_package():
    """Public reference methods remain available on the native CPU/reference backend."""
    _assert_import_smoke_case_passes(
        "case_optimizer_jax_public_reference_methods_work_without_private_package",
        "public optimizer_jax reference methods failed",
    )


def test_optimizer_jax_reference_methods_reject_all_jax_backend_modes():
    """Any JAX backend mode must reject host reference optimizer methods."""
    _assert_import_smoke_case_passes(
        "case_optimizer_jax_reference_methods_reject_all_jax_backend_modes",
        "JAX-backend reference optimizer guard failed",
    )


def test_optimizer_jax_private_methods_require_private_package_when_blocked():
    """Private optimizer methods must raise ImportError when the private package is absent."""
    _assert_import_smoke_case_passes(
        "case_optimizer_jax_private_methods_require_private_package_when_blocked",
        "private optimizer import guard failed",
    )


def test_optimizer_jax_private_nested_import_errors_propagate():
    """Nested private-package import failures must not be masked as package absence."""
    _assert_import_smoke_case_passes(
        "case_optimizer_jax_private_nested_import_errors_propagate",
        "nested private optimizer ImportError was masked",
    )


def test_optimizer_jax_public_module_has_no_private_jax_src_usage():
    """Section 6 public optimizer module must remain free of jax._src usage."""
    _assert_no_private_jax_src_usage(
        _OPTIMIZER_JAX_PATH,
        label="optimizer_jax.py in the public lane",
    )


def test_optimizer_jax_private_package_has_no_private_jax_src_usage():
    """Private optimizer modules must also stay on public JAX APIs."""
    forbidden_usages = {}
    for path in sorted(_OPTIMIZER_PRIVATE_DIR.glob("*.py")):
        usages = _find_private_jax_src_usages(path)
        if usages:
            forbidden_usages[str(path.relative_to(_OPTIMIZER_PRIVATE_DIR.parent))] = (
                usages
            )

    assert forbidden_usages == {}, (
        f"optimizer_jax_private must not use jax._src: {forbidden_usages}"
    )


def test_backend_runtime_module_has_no_private_jax_src_usage():
    """Backend runtime helpers must stay on public JAX APIs."""
    _assert_no_private_jax_src_usage(
        _RUNTIME_BACKEND_PATH,
        label="runtime.py in backend helpers",
    )


def test_jax_classes_inherit_optimizable():
    """JAX adapter classes use the real Optimizable metaclass."""
    _assert_import_smoke_case_passes(
        "case_jax_classes_inherit_optimizable",
        "inheritance check failed",
    )


def test_import_pure_jax_modules():
    """Pure JAX compute modules (M1) import through the package."""
    _assert_import_smoke_case_passes(
        "case_import_pure_jax_modules",
        "import pure JAX modules failed",
    )


def test_public_jax_helpers_are_exposed_on_package_roots():
    """simsopt.solve and simsopt.geo must expose the public JAX helper API."""
    _assert_import_smoke_case_passes(
        "case_public_jax_helpers_are_exposed_on_package_roots",
        "public JAX helper export surface failed",
    )


def test_m5_classes_require_simsoptpp():
    """M5 single-stage wrappers remain package-gated on simsoptpp availability."""
    _assert_import_smoke_case_passes(
        "case_m5_classes_require_simsoptpp",
        "M5 availability check failed",
    )


def test_direct_curve_modules_raise_clear_importerror_without_simsoptpp():
    """Direct geo-module imports should fail clearly at instantiation time."""
    _assert_import_smoke_case_passes(
        "case_direct_curve_modules_raise_clear_importerror_without_simsoptpp",
        "direct geo-module simsoptpp fallback smoke failed",
    )


def test_direct_optional_geo_modules_import_without_simsoptpp():
    """Optional geo modules should remain directly importable without simsoptpp."""
    _assert_import_smoke_case_passes(
        "case_direct_optional_geo_modules_import_without_simsoptpp",
        "optional geo-module import smoke failed",
    )


def test_curveobjectives_optional_cpp_helpers_raise_clear_importerror_without_simsoptpp():
    """Optional simsoptpp helpers in curveobjectives should fail clearly on use."""
    _assert_import_smoke_case_passes(
        "case_curveobjectives_optional_cpp_helpers_raise_clear_importerror_without_simsoptpp",
        "curveobjectives simsoptpp helper smoke failed",
    )


def test_framedcurve_direct_module_import_smoke():
    """Direct import of simsopt.geo.framedcurve should not hit jax_core cycles."""
    _assert_import_smoke_case_passes(
        "case_framedcurve_direct_module_import_smoke",
        "direct framedcurve import smoke failed",
    )


def test_biotsavart_jax_backend_does_not_import_coil_unwrap_helper():
    """The JAX backend must not depend on field/coil.py for graph unwrapping."""
    backend_path = Path(_SRC_DIR) / "simsopt" / "field" / "biotsavart_jax_backend.py"
    tree = ast.parse(backend_path.read_text(encoding="utf-8"))

    direct_coil_imports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "simsopt.field.coil" and node.module != "coil":
            continue
        imported_names = {alias.name for alias in node.names}
        if "_unwrap_coil_curve_and_current_objects" in imported_names:
            direct_coil_imports.append(node.lineno)

    assert not direct_coil_imports, (
        "biotsavart_jax_backend.py must not import "
        "_unwrap_coil_curve_and_current_objects from field/coil.py"
    )


def test_surfaceobjectives_jax_has_no_tensor_surface_imports():
    """Single-stage JAX wrappers should not instantiate tensor surfaces internally."""
    objectives_path = Path(_SRC_DIR) / "simsopt" / "geo" / "surfaceobjectives_jax.py"
    tree = ast.parse(objectives_path.read_text(encoding="utf-8"))

    tensor_surface_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        in (
            "simsopt.geo.surfacexyztensorfourier",
            "surfacexyztensorfourier",
        )
        and any(alias.name == "SurfaceXYZTensorFourier" for alias in node.names)
    ]

    assert not tensor_surface_import_lines, (
        "surfaceobjectives_jax.py must not import SurfaceXYZTensorFourier "
        "for its JAX wrapper/runtime helpers"
    )


def test_import_cpu_package_entrypoints_with_simsoptpp():
    """CPU package entrypoints must import cleanly when simsoptpp is available."""
    try:
        from simsoptpp import Curve as _  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    _assert_import_smoke_case_passes(
        "case_import_cpu_package_entrypoints_with_simsoptpp",
        "CPU entrypoint import check failed",
    )


def test_field_package_import_is_lazy_with_simsoptpp():
    """Bare package import must not eagerly load CPU field modules."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    _assert_import_smoke_case_passes(
        "case_field_package_import_is_lazy_with_simsoptpp",
        "field package import was not lazy",
    )


def test_geo_package_import_is_lazy_with_simsoptpp():
    """Bare package import must not eagerly load CPU geometry modules."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    _assert_import_smoke_case_passes(
        "case_geo_package_import_is_lazy_with_simsoptpp",
        "geo package import was not lazy",
    )


def test_import_cpu_geo_core_entrypoints_without_jax():
    """Core CPU geo entrypoints should import when simsoptpp is present but JAX is absent."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    _assert_import_smoke_case_passes(
        "case_import_cpu_geo_core_entrypoints_without_jax",
        "CPU geo import unexpectedly required jax",
    )
