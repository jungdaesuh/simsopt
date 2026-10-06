"""Multi-device sharding at extents that do not divide the device count.

Forced host devices stand in for several GPUs (``XLA_FLAGS`` acts before JAX
initializes, so each case runs ``tests/subprocess/uneven_sharding_cases.py``
in a fresh process). Three and four devices make most counts uneven: 4097
points, 3 or 5 coils, 7 surface rows, 7 seeds and 3 trajectory lanes.

Before padding, point sharding raised ``IndivisibleError`` for 4095 and 4097
points, coil arrays took the points' partition and failed for coil counts not
divisible by the device count, and zero-padded coil-collective coils gave NaN
at the origin in values and pullbacks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CASES = _REPO_ROOT / "tests" / "subprocess" / "uneven_sharding_cases.py"
_STRATEGIES = ("points", "coil_groups", "points_coils", "hybrid")


def _forced_device_environment(device_count: int) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("JAX_", "SIMSOPT_", "XLA_"))
    }
    environment.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={device_count}",
        PYTHONPATH=os.pathsep.join(
            path
            for path in (str(_REPO_ROOT / "src"), os.environ.get("PYTHONPATH"))
            if path
        ),
    )
    return environment


def _run_cases(device_count: int, *args: str) -> None:
    completed = subprocess.run(
        (sys.executable, str(_CASES), str(device_count), *args),
        env=_forced_device_environment(device_count),
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("device_count", [3, 4])
def test_biot_savart_jax_matches_native_and_single_device(device_count, strategy):
    """B, dB/dX, A, dA/dX, d2B/dXdX and the B/dB/A VJPs, finite at the origin."""
    _run_cases(device_count, "field", strategy)


@pytest.mark.parametrize("device_count", [3, 4])
def test_sharding_helper_consumers_match_unsharded(device_count):
    """Pairwise rows, surface quadrature, seed batches and trajectory batches."""
    _run_cases(device_count, "consumers")
