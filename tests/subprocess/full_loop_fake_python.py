#!/usr/bin/env python3
"""Tiny executable Python stand-in for full-loop orchestration tests."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import time


RECORDED_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "SLURMD_NODENAME",
    "SLURM_STEP_ID",
    "SLURM_STEP_NODELIST",
)


def _option_path(arguments: Sequence[str], option: str) -> Path | None:
    if option not in arguments:
        return None
    option_index = arguments.index(option)
    if option_index + 1 == len(arguments):
        raise ValueError(f"{option} requires a value")
    return Path(arguments[option_index + 1])


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    output_root = _option_path(arguments, "--output-root")
    run_dir = _option_path(arguments, "--run-dir")
    lane_root = output_root if output_root is not None else run_dir
    if lane_root is None:
        raise ValueError("driver invocation must contain --output-root or --run-dir")

    lane_root.mkdir(parents=True, exist_ok=True)
    observation = {
        "argv": list(arguments),
        "environment": {
            name: os.environ.get(name, "") for name in RECORDED_ENVIRONMENT_NAMES
        },
    }
    (lane_root / "driver_probe.json").write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    time.sleep(float(os.environ.get("FULL_LOOP_PROBE_HOLD_SECONDS", "0")))

    results_root = Path(os.environ["FULL_LOOP_PROBE_RESULTS_ROOT"])
    results_template = results_root / f"{lane_root.name}.json"
    shutil.copyfile(results_template, lane_root / "results.json")
    return int(os.environ.get("FULL_LOOP_PROBE_RETURN_CODE", "0"))


if __name__ == "__main__":
    raise SystemExit(main())
