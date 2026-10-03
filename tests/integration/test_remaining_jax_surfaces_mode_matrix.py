from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from simsopt_jax.backend import VALID_BACKEND_MODES


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
_BACKEND_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)

_SIMSOPTPP_FREE_IMPORT_PROBE = (
    Path(__file__).resolve().parents[1]
    / "subprocess"
    / "simsoptpp_free_jax_surfaces_probe.py"
)


def _mode_platform(mode: str) -> str:
    if mode.startswith("jax_gpu"):
        return "cuda"
    return "cpu"


def _jax_platform_available(platform: str) -> bool:
    if platform == "cpu":
        return True
    import jax

    try:
        return bool(jax.devices(platform))
    except RuntimeError:
        return False


def _mode_env(mode: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if key not in _BACKEND_ENV_VARS
    }
    env["PYTHONPATH"] = str(_SRC_DIR)
    env["SIMSOPT_BACKEND_MODE"] = mode
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if _mode_platform(mode) == "cuda":
        env["XLA_FLAGS"] = "--xla_gpu_exclude_nondeterministic_ops=true"
    return env


@pytest.mark.parametrize("mode", VALID_BACKEND_MODES)
def test_remaining_jax_surfaces_simsoptpp_free_import_mode_matrix(mode):
    platform = _mode_platform(mode)
    if not _jax_platform_available(platform):
        pytest.skip(f"JAX platform {platform!r} is not available in this environment")

    result = subprocess.run(
        [sys.executable, str(_SIMSOPTPP_FREE_IMPORT_PROBE)],
        cwd=_REPO_ROOT,
        env=_mode_env(mode),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["mode"] == mode
    assert payload["platform"] == platform
    assert payload["symbols"] == 2


def test_remaining_jax_surfaces_simsoptpp_backed_solve_exports():
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    import simsopt.solve as legacy_solve
    import simsopt_jax_adapters.solve.wireframe as wireframe_solve
    from simsopt_jax_adapters.solve.wireframe import (
        bnorm_obj_matrices_jax,
        get_gsco_iteration_jax,
        gsco_wireframe_jax,
        optimize_wireframe_jax,
        rcls_wireframe_jax,
        regularized_constrained_least_squares_jax,
    )

    adapter_exports = set(wireframe_solve.__all__)
    expected_exports = {
        "bnorm_obj_matrices_jax",
        "get_gsco_iteration_jax",
        "gsco_wireframe_jax",
        "optimize_wireframe_jax",
        "rcls_wireframe_jax",
        "regularized_constrained_least_squares_jax",
    }
    assert expected_exports <= adapter_exports
    assert expected_exports.isdisjoint(set(legacy_solve.__all__))
    assert all(
        symbol is not None
        for symbol in (
            bnorm_obj_matrices_jax,
            get_gsco_iteration_jax,
            gsco_wireframe_jax,
            optimize_wireframe_jax,
            rcls_wireframe_jax,
            regularized_constrained_least_squares_jax,
        )
    )
