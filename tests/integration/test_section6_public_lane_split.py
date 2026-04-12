"""Section 6 regression tests for the public/private optimizer split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip(
    "simsoptpp",
    reason="Section 6 real-fixture integration tests require simsoptpp",
)
_PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT),
        os.environ.get("PYTHONPATH", ""),
    ]
)


def _run_real_fixture_probe(
    *,
    optimizer_backend: str,
    block_private: bool,
    boozer_least_squares_algorithm: str | None = None,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = _PYTHONPATH
    script_path = REPO_ROOT / "tests" / "subprocess" / "section6_fixture_probe.py"
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--optimizer-backend",
        optimizer_backend,
    ]
    if block_private:
        cmd.append("--block-private")
    if boozer_least_squares_algorithm is not None:
        cmd.extend(["--boozer-least-squares-algorithm", boozer_least_squares_algorithm])
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "real-fixture subprocess failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "subprocess produced no output"
    return json.loads(lines[-1])


def _assert_public_scipy_lane_rejected(payload: dict) -> None:
    assert payload["exc_type"] == "ValueError"
    assert "boozer_optimizer_backend='ondevice'" in payload["exc_message"]
    assert "CPU/reference-only" in payload["exc_message"]


class TestSection6PublicLaneRealFixture:
    """Real Boozer parity regression for the Section 6 public/private split."""

    def test_public_scipy_lane_is_rejected_before_private_package_matters(self):
        baseline = _run_real_fixture_probe(
            optimizer_backend="scipy",
            block_private=False,
        )
        blocked = _run_real_fixture_probe(
            optimizer_backend="scipy",
            block_private=True,
        )

        for payload in (baseline, blocked):
            _assert_public_scipy_lane_rejected(payload)
        assert blocked == baseline

    @pytest.mark.private_optimizer_runtime
    def test_ondevice_lane_fails_on_real_fixture_when_private_package_is_blocked(self):
        blocked = _run_real_fixture_probe(
            optimizer_backend="ondevice",
            block_private=True,
            boozer_least_squares_algorithm="quasi-newton",
        )

        assert blocked["exc_type"] == "ImportError"
        assert "private optimizer package" in blocked["exc_message"]
        assert "simsopt.geo.optimizer_jax_private" in blocked["exc_message"]
