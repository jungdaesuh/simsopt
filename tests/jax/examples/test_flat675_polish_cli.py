"""The optional final check requires explicit design limits at the CLI boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/jax/3_Advanced/single_stage_flat675.py"
LIMITS = (
    "--max-boozer-rms",
    "0.01",
    "--max-label-error",
    "0.001",
    "--max-surface-movement",
    "0.02",
    "--max-objective-increase",
    "0.005",
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *arguments],
        cwd=ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_help_exposes_polish_and_its_explicit_acceptance_limits() -> None:
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    for option in ("--polish", "--bundle", "--smoke", *LIMITS[::2]):
        assert option in result.stdout


@pytest.mark.parametrize("arguments", [("--polish",), ("--polish", *LIMITS)])
def test_polish_cli_allows_unassessed_or_fully_specified_acceptance(
    arguments: tuple[str, ...],
) -> None:
    # Help validates arguments without launching the 661-DOF solve. Real
    # corrected-surface and native-oracle tests live with the polish adapter.
    result = _run(*arguments, "--help")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "arguments",
    [LIMITS, ("--polish", "--max-boozer-rms", "0.01")],
)
def test_partial_acceptance_contract_is_refused_before_solving(
    arguments: tuple[str, ...],
) -> None:
    result = _run(*arguments)
    assert result.returncode == 2
    assert "--polish and all four" in result.stderr


@pytest.mark.parametrize("invalid", ["nan", "inf", "-1"])
def test_nonfinite_or_negative_limits_cannot_disable_the_acceptance_gate(
    invalid: str,
) -> None:
    result = _run("--polish", "--max-boozer-rms", invalid, *LIMITS[2:])
    assert result.returncode == 2
    assert "finite and nonnegative" in result.stderr
