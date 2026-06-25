"""Measure dense d2B materialization against the private contracted helper."""

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
    parser.add_argument("--ncoils", type=int, default=12)
    parser.add_argument("--nquad", type=int, default=48)
    parser.add_argument("--npoints", type=int, default=64)
    parser.add_argument("--nleft", type=int, default=2)
    parser.add_argument("--nright", type=int, default=1)
    parser.add_argument("--coil-chunk-size", type=int, default=8)
    parser.add_argument("--quadrature-block-size", type=int, default=24)
    parser.add_argument("--point-chunk-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


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
        radius = 0.86 + 0.004 * coil_index
        z_scale = 0.06 + 0.002 * (coil_index % 5)
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
    radial_jitter = 0.015 * rng.normal(size=args.npoints)
    points = np.stack(
        (
            (0.25 + radial_jitter) * np.cos(point_theta),
            (0.25 + radial_jitter) * np.sin(point_theta),
            0.09 * np.sin(3.0 * point_theta),
        ),
        axis=1,
    )
    left_directions = rng.normal(size=(args.npoints, args.nleft, 3))
    right_directions = rng.normal(size=(args.npoints, args.nright, 3))
    currents = 1.0e5 + 5.0e3 * rng.normal(size=args.ncoils)
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(np.stack(gammas), dtype=jnp.float64),
        jnp.asarray(np.stack(gammadashs), dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
        jnp.asarray(left_directions, dtype=jnp.float64),
        jnp.asarray(right_directions, dtype=jnp.float64),
    )


def _measure_kernel(
    fn,
    inputs: tuple[jax.Array, ...],
    *,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
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
            compiled = fn.lower(*inputs).compile()
            compile_s = time.perf_counter() - compile_started_at
            execute_started_at = time.perf_counter()
            output = _block_ready(compiled(*inputs))
            first_execute_s = time.perf_counter() - execute_started_at
            for _ in range(warmup):
                _block_ready(compiled(*inputs))
            timings = []
            for _ in range(repeat):
                repeat_started_at = time.perf_counter()
                _block_ready(compiled(*inputs))
                timings.append(time.perf_counter() - repeat_started_at)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    peak_rss_mb = _peak_rss_mb()
    return {
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


def _json_case(case: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in case.items() if key != "output"}


def _contract_dense_output(
    dense_d2B,
    left_directions,
    right_directions,
):
    return jnp.einsum(
        "pjkl,paj,pbk->pabl",
        dense_d2B,
        left_directions,
        right_directions,
        precision=jax.lax.Precision.HIGHEST,
    )


def _output_delta(reference, actual) -> dict[str, float | int]:
    reference_array = np.asarray(reference)
    actual_array = np.asarray(actual)
    diff = np.abs(actual_array - reference_array)
    if diff.size:
        max_abs_diff = float(np.max(diff))
        denom = np.maximum(np.abs(reference_array), 1e-300)
        max_rel_diff = float(np.max(diff / denom))
    else:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
    return {
        "leaf_count": 1,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    points, gammas, gammadashs, currents, left_directions, right_directions = (
        _make_fixture(args)
    )
    core_bs.invalidate_kernel_cache()
    dense_kernel = core_bs._make_kernel(
        core_bs._Integrand.B,
        core_bs._DiffMode.HESSIAN,
        args.coil_chunk_size,
        args.quadrature_block_size,
        args.point_chunk_size,
        None,
    )
    contracted_kernel = core_bs._make_d2B_contracted_kernel(
        args.coil_chunk_size,
        args.quadrature_block_size,
        args.point_chunk_size,
    )
    dense = _measure_kernel(
        dense_kernel,
        (points, gammas, gammadashs, currents),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    contracted = _measure_kernel(
        contracted_kernel,
        (points, gammas, gammadashs, currents, left_directions, right_directions),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    expected = _contract_dense_output(
        dense["output"],
        left_directions,
        right_directions,
    )
    delta = _output_delta(expected, contracted["output"])
    dense_temp_size = int(dense["compiled_memory_analysis"]["temp_size_in_bytes"])
    contracted_temp_size = int(
        contracted["compiled_memory_analysis"]["temp_size_in_bytes"]
    )
    temp_savings = dense_temp_size - contracted_temp_size
    return {
        "measurement": "biotsavart phase-5 d2B contraction probe",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "shape": {
            "ncoils": int(args.ncoils),
            "nquad": int(args.nquad),
            "npoints": int(args.npoints),
            "nleft": int(args.nleft),
            "nright": int(args.nright),
        },
        "coil_chunk_size": int(args.coil_chunk_size),
        "quadrature_block_size": int(args.quadrature_block_size),
        "point_chunk_size": int(args.point_chunk_size),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "dense_hessian": _json_case(dense),
        "contracted_helper": _json_case(contracted),
        "contracted_minus_dense_hessian": {
            "compile_s": float(contracted["compile_s"]) - float(dense["compile_s"]),
            "post_compile_median_s": float(contracted["post_compile_median_s"])
            - float(dense["post_compile_median_s"]),
            "temp_size_in_bytes": contracted_temp_size - dense_temp_size,
        },
        "temp_savings_vs_dense_hessian": {
            "bytes": int(temp_savings),
            "fraction": float(temp_savings / dense_temp_size),
        },
        "output_delta_vs_dense_contraction": delta,
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
