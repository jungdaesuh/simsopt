"""Measure Biot-Savart quadrature block-size tradeoffs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])
bootstrap_local_simsopt(SRC_ROOT)

import jax
import jax.numpy as jnp

from simsopt_jax.core import biotsavart as core_bs


jax.config.update("jax_enable_x64", True)


_KERNELS = {
    "B": (core_bs._Integrand.B, core_bs._DiffMode.VALUE),
    "dB": (core_bs._Integrand.B, core_bs._DiffMode.JACOBIAN),
    "B_and_dB": (core_bs._Integrand.B, core_bs._DiffMode.VALUE_AND_JACOBIAN),
}


class _CompileCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling jit(" not in message:
            return
        self.count += 1
        self.messages.append(message.splitlines()[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ncoils", type=int, default=32)
    parser.add_argument("--nquad", type=int, default=128)
    parser.add_argument("--npoints", type=int, default=64)
    parser.add_argument("--coil-chunk-size", type=int, default=16)
    parser.add_argument("--point-chunk-size", type=int, default=0)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(0, 32, 64, 128))
    parser.add_argument("--kernels", default="B,dB,B_and_dB")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _parse_kernels(raw: str) -> tuple[str, ...]:
    kernels = tuple(field.strip() for field in raw.split(",") if field.strip())
    if not kernels:
        raise ValueError("--kernels must list at least one kernel.")
    unknown = tuple(kernel for kernel in kernels if kernel not in _KERNELS)
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}")
    return kernels


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _memory_analysis_summary(compiled) -> dict[str, int]:
    stats = compiled.memory_analysis()
    total_size = (
        stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        + stats.temp_size_in_bytes
        - stats.alias_size_in_bytes
    )
    return {
        "generated_code_size_in_bytes": int(stats.generated_code_size_in_bytes),
        "argument_size_in_bytes": int(stats.argument_size_in_bytes),
        "output_size_in_bytes": int(stats.output_size_in_bytes),
        "alias_size_in_bytes": int(stats.alias_size_in_bytes),
        "temp_size_in_bytes": int(stats.temp_size_in_bytes),
        "total_size_in_bytes": int(total_size),
        "host_generated_code_size_in_bytes": int(
            stats.host_generated_code_size_in_bytes
        ),
        "host_argument_size_in_bytes": int(stats.host_argument_size_in_bytes),
        "host_output_size_in_bytes": int(stats.host_output_size_in_bytes),
        "host_alias_size_in_bytes": int(stats.host_alias_size_in_bytes),
        "host_temp_size_in_bytes": int(stats.host_temp_size_in_bytes),
    }


def _block_ready(value):
    return jax.block_until_ready(value)


def _make_fixture(args: argparse.Namespace) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(args.seed)
    theta = np.linspace(0.0, 2.0 * np.pi, args.nquad, endpoint=False)
    gammas = []
    gammadashs = []
    for coil_index in range(args.ncoils):
        phase = 2.0 * np.pi * coil_index / max(args.ncoils, 1)
        radius = 0.9 + 0.003 * coil_index
        z_scale = 0.08 + 0.001 * (coil_index % 5)
        angle = theta + phase
        gammas.append(
            np.stack(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z_scale * np.sin(2.0 * angle),
                ),
                axis=1,
            )
        )
        gammadashs.append(
            np.stack(
                (
                    -radius * np.sin(angle),
                    radius * np.cos(angle),
                    2.0 * z_scale * np.cos(2.0 * angle),
                ),
                axis=1,
            )
        )
    point_theta = np.linspace(0.0, 2.0 * np.pi, args.npoints, endpoint=False)
    radial_jitter = 0.02 * rng.normal(size=args.npoints)
    points = np.stack(
        (
            (0.25 + radial_jitter) * np.cos(point_theta),
            (0.25 + radial_jitter) * np.sin(point_theta),
            0.11 * np.sin(3.0 * point_theta),
        ),
        axis=1,
    )
    currents = 1.0e5 + 1.0e4 * rng.normal(size=args.ncoils)
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(np.stack(gammas), dtype=jnp.float64),
        jnp.asarray(np.stack(gammadashs), dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _kernel_for(
    kernel_name: str,
    *,
    coil_chunk_size: int,
    quadrature_block_size: int,
    point_chunk_size: int,
):
    integrand, diff_mode = _KERNELS[kernel_name]
    return core_bs._make_kernel(
        integrand,
        diff_mode,
        coil_chunk_size,
        quadrature_block_size,
        point_chunk_size,
        None,
    )


def _measure_case(
    args: argparse.Namespace,
    *,
    kernel_name: str,
    quadrature_block_size: int,
    inputs: tuple[jax.Array, ...],
) -> dict[str, object]:
    points, gammas, gammadashs, currents = inputs
    kernel = _kernel_for(
        kernel_name,
        coil_chunk_size=args.coil_chunk_size,
        quadrature_block_size=quadrature_block_size,
        point_chunk_size=args.point_chunk_size,
    )
    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    rss_before_mb = _peak_rss_mb()
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            compile_started_at = time.perf_counter()
            compiled = kernel.lower(points, gammas, gammadashs, currents).compile()
            compile_s = time.perf_counter() - compile_started_at
            execute_started_at = time.perf_counter()
            output = _block_ready(compiled(points, gammas, gammadashs, currents))
            first_execute_s = time.perf_counter() - execute_started_at
            for _ in range(args.warmup):
                _block_ready(compiled(points, gammas, gammadashs, currents))
            timings = []
            for _ in range(args.repeat):
                repeat_started_at = time.perf_counter()
                _block_ready(compiled(points, gammas, gammadashs, currents))
                timings.append(time.perf_counter() - repeat_started_at)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    peak_rss_mb = _peak_rss_mb()
    return {
        "kernel": kernel_name,
        "quadrature_block_size": int(quadrature_block_size),
        "coil_chunk_size": int(args.coil_chunk_size),
        "point_chunk_size": int(args.point_chunk_size),
        "compile_log_count": int(handler.count),
        "compile_messages": handler.messages,
        "compile_s": float(compile_s),
        "first_execute_s": float(first_execute_s),
        "first_call_s": float(compile_s + first_execute_s),
        "post_compile_median_s": float(np.median(timings)),
        "post_compile_min_s": float(np.min(timings)),
        "post_compile_max_s": float(np.max(timings)),
        "rss_before_mb": float(rss_before_mb),
        "peak_rss_mb": float(peak_rss_mb),
        "peak_rss_delta_mb": float(max(0.0, peak_rss_mb - rss_before_mb)),
        "compiled_memory_analysis": _memory_analysis_summary(compiled),
        "output": output,
    }


def _output_delta(reference, actual) -> dict[str, float | int]:
    reference_leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(reference)]
    actual_leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(actual)]
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for reference_leaf, actual_leaf in zip(reference_leaves, actual_leaves, strict=True):
        diff = np.abs(actual_leaf - reference_leaf)
        if diff.size:
            max_abs_diff = max(max_abs_diff, float(np.max(diff)))
            denom = np.maximum(np.abs(reference_leaf), 1e-300)
            max_rel_diff = max(max_rel_diff, float(np.max(diff / denom)))
    return {
        "leaf_count": len(reference_leaves),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


def _json_case(case: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in case.items() if key != "output"}


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    inputs = _make_fixture(args)
    kernel_names = _parse_kernels(args.kernels)
    results = []
    for kernel_name in kernel_names:
        cases = [
            _measure_case(
                args,
                kernel_name=kernel_name,
                quadrature_block_size=block_size,
                inputs=inputs,
            )
            for block_size in args.block_sizes
        ]
        dense_output = cases[0]["output"]
        results.append(
            {
                "kernel": kernel_name,
                "cases": [
                    {
                        **_json_case(case),
                        "output_delta_vs_first_block": _output_delta(
                            dense_output,
                            case["output"],
                        ),
                    }
                    for case in cases
                ],
            }
        )
    return {
        "measurement": "biotsavart quadrature block-size probe",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "shape": {
            "ncoils": int(args.ncoils),
            "nquad": int(args.nquad),
            "npoints": int(args.npoints),
        },
        "coil_chunk_size": int(args.coil_chunk_size),
        "point_chunk_size": int(args.point_chunk_size),
        "block_sizes": list(args.block_sizes),
        "kernels": list(kernel_names),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
