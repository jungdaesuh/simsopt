"""Section 6 regression tests for the public/private optimizer split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

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
    code = f"""
        import importlib.abc
        import json
        import sys

        if {block_private!r}:
            class _BlockPrivateOptimizer(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if fullname == "simsopt.geo.optimizer_jax_private" or fullname.startswith(
                        "simsopt.geo.optimizer_jax_private."
                    ):
                        raise ImportError("blocked private optimizer package for section 6 test")
                    return None

            sys.meta_path.insert(0, _BlockPrivateOptimizer())

        from benchmarks.single_stage_smoke_fixture import build_real_single_stage_init_fixture
        from simsopt.geo import optimizer_jax

        payload = {{}}
        try:
            fixture = build_real_single_stage_init_fixture(
                backend="jax",
                optimizer_backend={optimizer_backend!r},
                boozer_least_squares_algorithm={boozer_least_squares_algorithm!r},
            )
        except Exception as exc:
            payload["exc_type"] = type(exc).__name__
            payload["exc_message"] = str(exc)
        else:
            booz = fixture["boozer_surface"]
            result = booz.res
            payload.update(
                {{
                    "success": bool(result["success"]),
                    "optimizer_method": result.get("optimizer_method"),
                    "iota": float(result["iota"]),
                    "G": float(result["G"]),
                    "fun": float(result["fun"]),
                    "iter": int(result["iter"]),
                    "surface_dofs": [float(x) for x in booz.surface.get_dofs()],
                    "private_pkg_is_none": optimizer_jax._private_pkg is None,
                    "private_in_sys_modules": "simsopt.geo.optimizer_jax_private" in sys.modules,
                }}
            )

        print(json.dumps(payload))
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
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
