# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax==0.9.2",
#     "jaxlib==0.9.2",
#     "numpy>=2.0",
#     "scipy>=1.13",
# ]
# ///
"""
End-to-end ``BoozerSurfaceJAX.run_code()`` CPU benchmark.

Usage:
    PYTHONPATH=src hf jobs uv run benchmarks/cpu_run_code_benchmark.py --flavor cpu-xl --timeout 15m

This benchmark requires a full repo environment with ``simsoptpp`` available.
On the public JAX ``0.9.2`` lane it defaults to ``optimizer_backend="scipy"``.
``ondevice`` and ``hybrid`` remain private-runtime backends that require the
separate JAX ``0.6.2`` optimizer lane.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
import jax

jax.config.update("jax_enable_x64", True)

from benchmarks.benchmark_config import available_config_labels, resolve_configs
from benchmarks.run_code_benchmark_common import (
    BENCHMARK_BACKEND_CHOICES,
    print_provenance,
    resolve_benchmark_backends,
    run_benchmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        choices=available_config_labels(),
        help="Benchmark config label to run. Repeat to run multiple configs.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BENCHMARK_BACKEND_CHOICES,
        help="Optimizer backend to benchmark. Repeat to run multiple backends.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of fresh-solve repeats per backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = resolve_benchmark_backends(args.backend)
    print_provenance("JAX run_code() CPU Benchmark", backends)
    run_benchmarks(
        title="JAX run_code() CPU Benchmark",
        configs=resolve_configs(args.config),
        backends=backends,
        repeats=args.repeats,
    )
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
