"""Compile-time memory analysis for pairwise thresholded reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])
bootstrap_local_simsopt(SRC_ROOT)

import jax
import jax.numpy as jnp

from simsopt.geo._pairwise_reductions import (
    pairwise_thresholded_mean_square_distance_pure,
)


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile gradient kernels for the pairwise thresholded mean-square "
            "distance reducer and report XLA memory_analysis() sizes."
        )
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rows", type=int, default=16320)
    parser.add_argument("--cols", type=int, default=16320)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=(0, 256))
    parser.add_argument("--minimum-distance", type=float, default=0.5)
    parser.add_argument("--seed-a", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def pairwise_inputs(args: argparse.Namespace) -> tuple[jax.Array, jax.Array]:
    points_a = jax.random.normal(
        jax.random.PRNGKey(args.seed_a),
        (args.rows, 3),
        dtype=jnp.float64,
    )
    points_b = jax.random.normal(
        jax.random.PRNGKey(args.seed_b),
        (args.cols, 3),
        dtype=jnp.float64,
    )
    return points_a, points_b


def memory_analysis_for_chunk_size(
    args: argparse.Namespace,
    points_a: jax.Array,
    points_b: jax.Array,
    chunk_size: int,
) -> dict[str, int | float]:
    def objective(current_points_a, current_points_b):
        return pairwise_thresholded_mean_square_distance_pure(
            current_points_a,
            current_points_b,
            args.minimum_distance,
            chunk_size=chunk_size,
        )

    compiled = (
        jax.jit(jax.grad(objective, argnums=0))
        .lower(points_a, points_b)
        .compile()
    )
    stats = compiled.memory_analysis()
    total_size_in_bytes = (
        stats.temp_size_in_bytes
        + stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        - stats.alias_size_in_bytes
    )
    return {
        "rows": args.rows,
        "cols": args.cols,
        "chunk_size": chunk_size,
        "minimum_distance": args.minimum_distance,
        "temp_size_in_bytes": stats.temp_size_in_bytes,
        "argument_size_in_bytes": stats.argument_size_in_bytes,
        "output_size_in_bytes": stats.output_size_in_bytes,
        "alias_size_in_bytes": stats.alias_size_in_bytes,
        "total_size_in_bytes": total_size_in_bytes,
        "temp_size_in_mib": stats.temp_size_in_bytes / (1024**2),
        "total_size_in_mib": total_size_in_bytes / (1024**2),
    }


def main() -> None:
    args = parse_args()
    points_a, points_b = pairwise_inputs(args)
    results = [
        memory_analysis_for_chunk_size(args, points_a, points_b, chunk_size)
        for chunk_size in args.chunk_sizes
    ]
    payload = {"measurements": results}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.write_text(f"{text}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
