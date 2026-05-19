from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _build_env(repo_root: Path, *, sharding: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_ENABLE_X64"] = "True"
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    env["SIMSOPT_JAX_SHARDING"] = sharding
    env["SIMSOPT_JAX_MIN_POINTS_TO_SHARD"] = "1"
    env["SIMSOPT_JAX_TRANSFER_GUARD"] = "allow"
    return env


@pytest.mark.parametrize(
    ("case", "sharding"),
    [
        ("surface-quadrature-sharding", "points"),
        ("seed-batch-value-grad-sharding", "points"),
        ("target-minimize-replicated-sharding-vjp", "points"),
        ("target-minimize-replicated-sharding-vjp", "points_coils"),
        ("boozer-penalty-replicated-sharding-vjp", "points"),
        ("boozer-penalty-replicated-sharding-vjp", "points_coils"),
    ],
)
def test_forced_cpu_point_axis_sharding_cases(case: str, sharding: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tests" / "subprocess" / "jax_runtime_cases.py"

    result = subprocess.run(
        (sys.executable, str(script), case),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=_build_env(repo_root, sharding=sharding),
    )

    assert result.returncode == 0, result.stdout + result.stderr
