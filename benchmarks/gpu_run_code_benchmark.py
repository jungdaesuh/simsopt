# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax[cuda12]",
#     "numpy",
#     "scipy",
# ]
# ///
"""
End-to-end ``BoozerSurfaceJAX.run_code()`` GPU benchmark.

Usage:
    PYTHONPATH=src hf jobs uv run benchmarks/gpu_run_code_benchmark.py --flavor a100-large --timeout 15m

This benchmark requires a full repo environment with ``simsoptpp`` available.
It compares the least-squares inner-solver backends on the same run_code path:
``optimizer_backend=scipy``, ``ondevice``, and ``hybrid``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax

jax.config.update("jax_enable_x64", True)

from benchmarks.run_code_benchmark_common import print_provenance, run_benchmarks


def main() -> None:
    print_provenance("JAX run_code() GPU Benchmark")
    run_benchmarks(title="JAX run_code() GPU Benchmark")
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
