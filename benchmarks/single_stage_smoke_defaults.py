"""Lightweight shared defaults for the real single-stage smoke fixtures."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp22ginsburg_000_014417_iota15.nc"
# Vendored strict-cap Stage 2 donor for the real single-stage smoke/parity probes.
DEFAULT_STAGE2_SEED_DIR = (
    REPO_ROOT / "benchmarks" / "fixtures" / "single_stage_seed_iota15"
)
DEFAULT_STAGE2_BS_PATH = DEFAULT_STAGE2_SEED_DIR / "biot_savart_opt.json"
DEFAULT_STAGE2_RESULTS_PATH = DEFAULT_STAGE2_SEED_DIR / "results.json"
DEFAULT_STAGE2_BS_REL_PATH = DEFAULT_STAGE2_BS_PATH.relative_to(REPO_ROOT)
