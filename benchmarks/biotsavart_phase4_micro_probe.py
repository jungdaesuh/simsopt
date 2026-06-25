"""Phase-4 Biot-Savart micro-experiments for remat and r_inv3 algebra."""

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

    def emit(self, record: logging.LogRecord) -> None:
        if "Compiling jit(" in record.getMessage():
            self.count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ncoils", type=int, default=16)
    parser.add_argument("--nquad", type=int, default=64)
    parser.add_argument("--npoints", type=int, default=256)
    parser.add_argument("--coil-chunk-size", type=int, default=16)
    parser.add_argument("--quadrature-block-size", type=int, default=32)
    parser.add_argument("--point-chunk-sizes", type=int, nargs="+", default=(16, 64, 256))
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
        "argument_size_in_bytes": int(stats.argument_size_in_bytes),
        "output_size_in_bytes": int(stats.output_size_in_bytes),
        "alias_size_in_bytes": int(stats.alias_size_in_bytes),
        "temp_size_in_bytes": int(stats.temp_size_in_bytes),
        "total_size_in_bytes": int(total_size),
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
        radius = 0.85 + 0.004 * coil_index
        z_scale = 0.08 + 0.002 * (coil_index % 7)
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
            (0.24 + radial_jitter) * np.cos(point_theta),
            (0.24 + radial_jitter) * np.sin(point_theta),
            0.10 * np.sin(3.0 * point_theta),
        ),
        axis=1,
    )
    currents = 1.0e5 + 5.0e3 * rng.normal(size=args.ncoils)
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(np.stack(gammas), dtype=jnp.float64),
        jnp.asarray(np.stack(gammadashs), dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
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


def _remat_cases(
    args: argparse.Namespace,
    inputs: tuple[jax.Array, ...],
) -> list[dict[str, object]]:
    cases = []
    original_checkpoint = core_bs.jax.checkpoint
    for point_chunk_size in args.point_chunk_sizes:
        for remat_enabled in (True, False):
            core_bs.invalidate_kernel_cache()
            core_bs.jax.checkpoint = original_checkpoint if remat_enabled else lambda f: f
            try:
                kernel = core_bs._make_kernel(
                    core_bs._Integrand.B,
                    core_bs._DiffMode.VALUE_AND_JACOBIAN,
                    args.coil_chunk_size,
                    args.quadrature_block_size,
                    point_chunk_size,
                    None,
                )

                def objective(points, gammas, gammadashs, currents):
                    B, dB = kernel(points, gammas, gammadashs, currents)
                    return jnp.sum(B) + 0.01 * jnp.sum(dB)

                grad_kernel = jax.jit(jax.grad(objective, argnums=0))
                measurement = _measure_kernel(
                    grad_kernel,
                    inputs,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
            finally:
                core_bs.jax.checkpoint = original_checkpoint
                core_bs.invalidate_kernel_cache()
            cases.append(
                {
                    "point_chunk_size": int(point_chunk_size),
                    "remat_enabled": bool(remat_enabled),
                    "measured_kernel": "grad_points_sum_B_plus_0p01_sum_dB",
                    **_json_case(measurement),
                }
            )
    return cases


def _triple_r_inv3_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = core_bs._radius_squared(diff)
    r_inv = core_bs._explicit_rsqrt(r2)
    r_inv3 = r_inv * r_inv * r_inv
    cross = core_bs._cross_product(diff, gammadashs)
    return cross * r_inv3[..., None]


def _B_kernel_with_integrand(integrand):
    @jax.jit
    def kernel(points, gammas, gammadashs, currents):
        def per_point(x):
            values = integrand(x, gammas, gammadashs)
            integral = core_bs._pairwise_sum_axis(values, axis=1) / gammas.shape[1]
            return core_bs._float64_scalar(currents, core_bs._MU0_OVER_4PI) * jnp.einsum(
                "c,cj->j",
                currents,
                integral,
                precision=jax.lax.Precision.HIGHEST,
            )

        return jax.vmap(per_point, in_axes=(0,))(points)

    return kernel


def _r_inv3_cases(
    args: argparse.Namespace,
    inputs: tuple[jax.Array, ...],
) -> dict[str, object]:
    current = _measure_kernel(
        _B_kernel_with_integrand(core_bs._biot_savart_B_integrand),
        inputs,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    triple = _measure_kernel(
        _B_kernel_with_integrand(_triple_r_inv3_integrand),
        inputs,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    return {
        "current_r_inv_times_inv_r2": _json_case(current),
        "triple_r_inv": _json_case(triple),
        "triple_minus_current": {
            "compile_s": float(triple["compile_s"]) - float(current["compile_s"]),
            "post_compile_median_s": float(triple["post_compile_median_s"])
            - float(current["post_compile_median_s"]),
            "temp_size_in_bytes": int(
                triple["compiled_memory_analysis"]["temp_size_in_bytes"]
            )
            - int(current["compiled_memory_analysis"]["temp_size_in_bytes"]),
        },
        "output_delta": _output_delta(current["output"], triple["output"]),
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    inputs = _make_fixture(args)
    return {
        "measurement": "biotsavart phase-4 remat and r_inv3 micro-probe",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "shape": {
            "ncoils": int(args.ncoils),
            "nquad": int(args.nquad),
            "npoints": int(args.npoints),
        },
        "coil_chunk_size": int(args.coil_chunk_size),
        "quadrature_block_size": int(args.quadrature_block_size),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "remat_cases": _remat_cases(args, inputs),
        "r_inv3_cases": _r_inv3_cases(args, inputs),
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
