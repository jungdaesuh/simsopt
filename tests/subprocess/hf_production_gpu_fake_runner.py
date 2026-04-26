#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Sequence


def _load_config(script_path: Path) -> dict[str, object]:
    config_path = script_path.resolve().parents[1] / "fake_proof_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _output_json_path(argv: Sequence[str]) -> Path:
    for index, token in enumerate(argv):
        if token == "--output-json":
            return Path(argv[index + 1])
    raise SystemExit("missing --output-json")


def _append_call_record(
    config: dict[str, object],
    argv: Sequence[str],
    output_json: Path,
) -> None:
    call_log = Path(str(config["call_log"]))
    call_log.parent.mkdir(parents=True, exist_ok=True)
    with call_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "argv": list(argv),
                    "output_json": str(output_json),
                    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                    "cuda_library_mode": os.environ.get(
                        "SIMSOPT_JAX_CUDA_LIBRARY_MODE"
                    ),
                    "xla_flags": os.environ.get("XLA_FLAGS"),
                    "jax_compilation_cache_dir": os.environ.get(
                        "JAX_COMPILATION_CACHE_DIR"
                    ),
                    "xla_python_client_preallocate": os.environ.get(
                        "XLA_PYTHON_CLIENT_PREALLOCATE"
                    ),
                }
            )
            + "\n"
        )


def _proof_parity(script_path: Path, *, is_stage2: bool) -> dict[str, object]:
    repo_root = script_path.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from benchmarks.validation_ladder_contract import gpu_proof_parity_contract

    contract = gpu_proof_parity_contract("stage2" if is_stage2 else "single_stage")
    return {
        **contract,
        "cpu_oracle_value": 1.0,
        "gpu_value": 1.0,
        "value_rel_diff": 0.0,
        "gradient_rel_diff": 0.0 if is_stage2 else None,
    }


def _write_payload(
    output_json: Path,
    *,
    elapsed_s: float,
    script_path: Path,
    is_stage2: bool,
    invalid_proof_rtol: bool,
    invalid_value_rel_diff: bool,
    invalid_gradient_rel_diff: bool,
) -> None:
    proof_parity = _proof_parity(script_path, is_stage2=is_stage2)
    if invalid_proof_rtol:
        proof_parity["value_rtol"] = 1.0
    if invalid_value_rel_diff:
        proof_parity["value_rel_diff"] = 1.0
    if invalid_gradient_rel_diff:
        proof_parity["gradient_rel_diff"] = 1.0
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "passed": True,
                "elapsed_s": elapsed_s,
                "failures": [],
                "proof_parity": proof_parity,
                "bundle_provenance": {
                    "runner": "tests/subprocess/hf_production_gpu_fake_runner.py",
                    "fake": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _is_invalid_stage2_cold_mode(
    *,
    is_stage2: bool,
    output_json: Path,
    config: dict[str, object],
    mode: str,
) -> bool:
    return (
        is_stage2
        and output_json.name == "stage2_cold.json"
        and str(config["stage2_warm_mode"]) == mode
    )


def main(argv: Sequence[str] | None = None) -> int:
    script_path = Path(__file__)
    args = list(sys.argv[1:] if argv is None else argv)
    output_json = _output_json_path(args)
    config = _load_config(script_path)
    _append_call_record(config, args, output_json)

    is_stage2 = script_path.name == "stage2_e2e_comparison.py"
    if is_stage2 and output_json.name == "stage2_warm.json":
        stage2_warm_mode = str(config["stage2_warm_mode"])
        if stage2_warm_mode == "missing":
            return 3
        if stage2_warm_mode == "corrupt":
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text("{bad json", encoding="utf-8")
            return 3

    _write_payload(
        output_json,
        elapsed_s=1.0 if is_stage2 else 2.0,
        script_path=script_path,
        is_stage2=is_stage2,
        invalid_proof_rtol=_is_invalid_stage2_cold_mode(
            is_stage2=is_stage2,
            output_json=output_json,
            config=config,
            mode="invalid_proof_rtol",
        ),
        invalid_value_rel_diff=_is_invalid_stage2_cold_mode(
            is_stage2=is_stage2,
            output_json=output_json,
            config=config,
            mode="invalid_value_rel_diff",
        ),
        invalid_gradient_rel_diff=_is_invalid_stage2_cold_mode(
            is_stage2=is_stage2,
            output_json=output_json,
            config=config,
            mode="invalid_gradient_rel_diff",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
