from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from simsopt.backend import VALID_BACKEND_MODES


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

_SIMSOPTPP_FREE_IMPORT_PROGRAM = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import json
    import os
    import sys

    class BlockSimsoptpp(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "simsoptpp" or fullname.startswith("simsoptpp."):
                raise ModuleNotFoundError("blocked simsoptpp for mode-matrix smoke")
            return None

    sys.meta_path.insert(0, BlockSimsoptpp())

    import simsopt
    import simsopt.solve as solve
    from simsopt.backend import get_backend_mode, get_jax_platform
    from simsopt.mhd import (
        ProfilePolynomialJAX,
        ProfilePressureJAX,
        ProfileScaledJAX,
        ProfileSplineJAX,
        RedlBootstrapJAX,
        RedlDetailsJAX,
        VmecFrozenSplineState,
        VmecGeometryResultsJAX,
        compute_trapped_fraction_jax,
        j_dot_B_Redl_jax,
        j_dot_B_Redl_jax_from_arrays,
        vmec_compute_geometry_jax,
        vmec_fieldlines_jax,
        vmec_freeze_splines,
    )
    from simsopt.solve import (
        TraceableLeastSquaresProblem,
        least_squares_serial_solve_jax,
        traceable_least_squares_mpi_jacobian,
    )

    direct_modules = {
        "simsopt.solve.serial_jax": (
            "TraceableEqualityConstrainedProblem",
            "TraceableLeastSquaresProblem",
            "TraceableScalarProblem",
            "constrained_serial_solve_jax",
            "least_squares_serial_solve_jax",
            "serial_solve_jax",
            "traceable_least_squares_jacobian",
        ),
        "simsopt.solve.mpi_jax": (
            "least_squares_mpi_solve_jax",
            "traceable_least_squares_mpi_jacobian",
        ),
        "simsopt.solve.permanent_magnet_optimization_jax": (
            "GPMO_ArbVec_backtracking_jax",
            "GPMO_ArbVec_jax",
            "GPMO_backtracking_jax",
            "GPMO_baseline_jax",
            "GPMO_multi_jax",
            "relax_and_split_jax",
        ),
        "simsopt.jax_core._finite_difference": (
            "forward_jacobian_shard_map",
            "forward_jacobian_shard_map_columns",
            "forward_jacobian_vmap",
        ),
        "simsopt.jax_core.mhd_bootstrap": ("compute_trapped_fraction_jax",),
        "simsopt.jax_core.profiles": (
            "profile_polynomial_dfds",
            "profile_polynomial_value",
            "profile_pressure_dfds",
            "profile_pressure_value",
            "profile_scaled_dfds",
            "profile_scaled_value",
            "profile_spline_dfds",
            "profile_spline_value",
        ),
        "simsopt.jax_core.redl_current": (
            "RedlDetailsJAX",
            "j_dot_B_Redl_jax_from_arrays",
        ),
        "simsopt.jax_core.vmec_fieldlines": (
            "theta_vmec_from_theta_pest_implicit_jax",
            "theta_vmec_from_theta_pest_scan_jax",
            "theta_vmec_residual_jax",
        ),
        "simsopt.jax_core.vmec_geometry": (
            "VmecGeometryResultsJAX",
            "vmec_compute_geometry_jax",
        ),
    }
    for module_name, symbols in direct_modules.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            getattr(module, symbol)

    public_symbols = (
        ProfilePolynomialJAX,
        ProfilePressureJAX,
        ProfileScaledJAX,
        ProfileSplineJAX,
        RedlBootstrapJAX,
        RedlDetailsJAX,
        VmecFrozenSplineState,
        VmecGeometryResultsJAX,
        compute_trapped_fraction_jax,
        j_dot_B_Redl_jax,
        j_dot_B_Redl_jax_from_arrays,
        vmec_compute_geometry_jax,
        vmec_fieldlines_jax,
        vmec_freeze_splines,
        TraceableLeastSquaresProblem,
        least_squares_serial_solve_jax,
        traceable_least_squares_mpi_jacobian,
    )
    assert all(symbol is not None for symbol in public_symbols)
    assert "optimize_wireframe_jax" not in solve.__all__
    assert "simsoptpp" not in sys.modules
    print(json.dumps({
        "mode": get_backend_mode(),
        "platform": get_jax_platform(),
        "simsopt_file": simsopt.__file__,
        "symbols": len(public_symbols),
    }, sort_keys=True))
    """
)


def _mode_platform(mode: str) -> str:
    if mode.startswith("jax_gpu"):
        return "cuda"
    if mode == "jax_mps_smoke":
        return "mps"
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
        [sys.executable, "-c", _SIMSOPTPP_FREE_IMPORT_PROGRAM],
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
    assert payload["symbols"] == 17


def test_remaining_jax_surfaces_simsoptpp_backed_solve_exports():
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    import simsopt.solve as solve
    from simsopt.solve import (
        bnorm_obj_matrices_jax,
        get_gsco_iteration_jax,
        gsco_wireframe_jax,
        optimize_wireframe_jax,
        rcls_wireframe_jax,
        regularized_constrained_least_squares_jax,
    )

    solve_exports = set(solve.__all__)
    assert {
        "bnorm_obj_matrices_jax",
        "get_gsco_iteration_jax",
        "gsco_wireframe_jax",
        "optimize_wireframe_jax",
        "rcls_wireframe_jax",
        "regularized_constrained_least_squares_jax",
    } <= solve_exports
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
