"""Lightweight shared defaults for the real single-stage smoke fixtures."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp22ginsburg_000_014417_iota15.nc"
# Vendored strict-cap Stage 2 donor for the real single-stage smoke/parity probes.
SMOKE_TEST_STAGE2_SEED_DIR = (
    REPO_ROOT / "benchmarks" / "fixtures" / "single_stage_seed_iota15"
)
SMOKE_TEST_STAGE2_BS_PATH = SMOKE_TEST_STAGE2_SEED_DIR / "biot_savart_opt.json"
SMOKE_TEST_STAGE2_RESULTS_PATH = SMOKE_TEST_STAGE2_SEED_DIR / "results.json"
SMOKE_TEST_STAGE2_RUNTIME_SPEC_PATH = (
    SMOKE_TEST_STAGE2_SEED_DIR / "single_stage_jax_runtime_spec.json"
)
SMOKE_TEST_STAGE2_BS_REL_PATH = SMOKE_TEST_STAGE2_BS_PATH.relative_to(REPO_ROOT)
