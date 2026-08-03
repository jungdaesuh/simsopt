"""CPU-only contract tests for the prepared L-BFGS warm soak."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import lbfgs_warm_soak as soak


def _cpu_run(
    *,
    run_index: int,
    rss_kib: int,
    warm_seconds: float = 1.0,
    executable_count: int = 1,
) -> soak.WarmRunRecord:
    return soak.WarmRunRecord(
        run_index=run_index,
        retained=True,
        warm_seconds=warm_seconds,
        rss_kib=rss_kib,
        vram=soak.VramRecord(
            availability="unavailable",
            reason="cpu-device",
            vram_mib=None,
        ),
        executable_count=executable_count,
    )


def test_cpu_coil47_soak_emits_plateau_artifact(tmp_path: Path) -> None:
    output_json = tmp_path / "lbfgs-warm-soak.json"

    artifact = soak.run_soak(
        device="cpu",
        fixture_name="coil47",
        runs=5,
        output_json=output_json,
        maxiter=2,
        maxcor=3,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["config"] == {
        "device": "cpu",
        "discarded_runs": 1,
        "fixture": "coil47",
        "maxcor": 3,
        "maxiter": 2,
        "method": "lbfgs",
        "provider": "custom",
        "retained_runs": 4,
        "run_mode": "fused_stepwise",
        "runs": 5,
    }
    assert len(payload["runs"]) == 5
    retained_runs = [record for record in payload["runs"] if record["retained"]]
    assert len(retained_runs) == 4
    assert len({record["executable_count"] for record in retained_runs}) == 1
    assert retained_runs[0]["executable_count"] > 0
    assert all(
        record["vram"]
        == {
            "availability": "unavailable",
            "reason": "cpu-device",
            "vram_mib": None,
        }
        for record in payload["runs"]
    )
    assert payload["plateau_verdict"]["executable_count_identical"] is True
    assert payload["plateau_verdict"]["executable_count_positive"] is True
    assert payload["plateau_verdict"]["vram_applicable"] is False
    assert payload["plateau_verdict"]["vram_plateau"] is None
    assert payload["plateau_verdict"]["plateau"] is True
    assert artifact.plateau_verdict.plateau is True


def test_rss_growth_fails_plateau_verdict() -> None:
    records = (
        _cpu_run(run_index=1, rss_kib=10_000),
        _cpu_run(run_index=2, rss_kib=10_100),
        _cpu_run(
            run_index=3,
            rss_kib=10_100 + soak.RSS_PLATEAU_SLACK_KIB + 1,
        ),
        _cpu_run(run_index=4, rss_kib=10_200),
    )

    verdict = soak.compute_plateau_verdict(records, vram_applicable=False)

    assert verdict.rss_plateau is False
    assert verdict.plateau is False


def test_missing_pjit_cache_size_fails_closed() -> None:
    with pytest.raises(
        TypeError,
        match=r"does not expose PjitFunction\._cache_size\(\)",
    ):
        soak._executable_count(object())


def test_zero_executable_count_fails_plateau_verdict() -> None:
    records = tuple(
        _cpu_run(run_index=run_index, rss_kib=10_000, executable_count=0)
        for run_index in range(1, 5)
    )

    verdict = soak.compute_plateau_verdict(records, vram_applicable=False)

    assert verdict.executable_count_identical is True
    assert verdict.executable_count_positive is False
    assert verdict.plateau is False


def test_cpu_device_label_rejects_a_gpu_backend() -> None:
    with pytest.raises(RuntimeError, match="requested CPU warm soak"):
        soak._validate_device_binding("cpu", backend="cuda", platform="cuda")
