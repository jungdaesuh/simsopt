from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _build_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_ENABLE_X64"] = "True"
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    env["SIMSOPT_JAX_SHARDING"] = "points_coils"
    env["SIMSOPT_JAX_MIN_COILS_TO_SHARD"] = "1"
    env["SIMSOPT_JAX_TRANSFER_GUARD"] = "disallow"
    return env


@pytest.mark.parametrize(
    "case",
    [
        "grouped-points-coils-collective",
        "grouped-points-coils-non-divisible",
    ],
)
def test_points_coils_forced_cpu_stablehlo_all_reduce(case: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tests" / "subprocess" / "jax_runtime_cases.py"
    env = _build_env(repo_root)

    result = subprocess.run(
        (sys.executable, str(script), case),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
