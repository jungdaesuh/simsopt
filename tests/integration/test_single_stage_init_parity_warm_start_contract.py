"""Warm-start donor contract for high-resolution single-stage parity runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from benchmarks.single_stage_init_parity import (
    _resolve_warm_start_seed_contract_G,
    _validate_warm_start_seed_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "benchmarks" / "fixtures" / "single_stage_seed_iota15"


def test_warm_start_contract_accepts_runtime_g_derived_from_biotsavart(tmp_path):
    """Legacy donors without ``FINAL_G`` remain valid when TF currents define G."""
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    shutil.copyfile(FIXTURE / "biot_savart_opt.json", seed_dir / "biot_savart_opt.json")
    results = {
        "init_only": False,
        "FINAL_IOTA": 0.22650585872006085,
        "HARDWARE_CONSTRAINTS_OK": True,
        "SELF_INTERSECTING": False,
    }
    (seed_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    args = argparse.Namespace(iota_target=0.22650585872006085, num_tf_coils=20)

    derived_g = _resolve_warm_start_seed_contract_G(args, seed_dir, results)

    assert math.isfinite(float(derived_g))
    _validate_warm_start_seed_contract(args, seed_dir)
