from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import run_single_stage_fullspace_gpu as runner
from benchmarks.single_stage_fullspace_receipt import SCHEMA_VERSION_V3, RunRequest
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--phase",
        "canary",
        "--route",
        "CFS-FTR1",
        "--device",
        "rtx5090",
        "--steps",
        "10",
        "--output",
        str(tmp_path / "campaign"),
        *extra,
    ]


def test_ftr_preflight_is_v3_and_allows_existing_output(
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    (tmp_path / "campaign").mkdir()

    assert runner.main(_argv(tmp_path, "--preflight-only")) == 0

    payload = json.loads(capsysbinary.readouterr().out)
    assert payload["schema_version"] == SCHEMA_VERSION_V3
    assert payload["request"]["route"] == "CFS-FTR1"
    assert payload["request"]["steps"] == 10


def test_ftr_main_dispatches_without_prior_route_fallthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    observed: list[RunRequest] = []

    def fake_ftr(request: RunRequest, *_args: object, **_kwargs: object) -> bytes:
        observed.append(request)
        return b'{"route":"CFS-FTR1"}'

    monkeypatch.setattr(runner, "run_cfs_ftr1_campaign", fake_ftr)
    monkeypatch.setattr(runner.simsoptpp, "__file__", "/tmp/simsoptpp.so")
    monkeypatch.setattr(
        runner,
        "run_cfs_sqp1_campaign",
        lambda *_args, **_kwargs: pytest.fail("FTR fell through to CFS-SQP1"),
    )

    assert runner.main(_argv(tmp_path)) == 0

    assert observed[0].route is FullSpaceRoute.CFS_FTR1
    assert capsysbinary.readouterr().out == b'{"route":"CFS-FTR1"}'


@pytest.mark.parametrize(
    "replacement",
    (
        ("--steps", "100"),
        ("--device", "a100"),
    ),
)
def test_ftr_parser_rejects_non_gate2_request(
    tmp_path: Path, replacement: tuple[str, str]
) -> None:
    argv = _argv(tmp_path)
    index = argv.index(replacement[0]) + 1
    argv[index] = replacement[1]

    with pytest.raises(ValueError, match="RTX 5090 ten-step"):
        runner.parse_request(argv)


def test_ftr_probe_rejects_non_gpu_runtime_before_building_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "cpu")

    with pytest.raises(ValueError, match="exactly one JAX GPU"):
        runner.run_cfs_ftr1_probe(object())
