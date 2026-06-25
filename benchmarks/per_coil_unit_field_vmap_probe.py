"""Scaling probe for per-coil unit-field vectorization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import resource
import subprocess
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

from simsopt_jax_adapters.field.biotsavart_backend import (
    _per_coil_unit_field,
    _per_coil_unit_field_with_batch_size,
)
from simsopt_jax.backend import get_field_kernel_tuning
from simsopt_jax.core.biotsavart import (
    biot_savart_B,
    biot_savart_d2B_by_dXdX,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec


jax.config.update("jax_enable_x64", True)

_MAPPING_MODES = ("unbounded_vmap", "bounded_lax_map")
_KERNELS = {
    "B": biot_savart_B,
    "dB": biot_savart_dB_by_dX,
    "d2B": biot_savart_d2B_by_dXdX,
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
    parser.add_argument("--ncoils", default="2,4,8,16")
    parser.add_argument("--npoints", type=int, default=256)
    parser.add_argument("--nquad", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--kernels", default="B")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compare-vmap-batching", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument(
        "--child-mapping-mode",
        choices=_MAPPING_MODES,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--child-kernel",
        choices=tuple(_KERNELS),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child-ncoils", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _parse_ncoils(raw: str) -> tuple[int, ...]:
    values = tuple(int(field) for field in raw.split(",") if field.strip())
    if len(values) < 2:
        raise ValueError("--ncoils must list at least two positive sizes.")
    if any(value <= 0 for value in values):
        raise ValueError("--ncoils entries must be positive.")
    return values


def _parse_kernels(raw: str) -> tuple[str, ...]:
    values = tuple(field.strip() for field in raw.split(",") if field.strip())
    if not values:
        raise ValueError("--kernels must list at least one kernel.")
    unknown = tuple(value for value in values if value not in _KERNELS)
    if unknown:
        raise ValueError(f"unknown --kernels entries: {unknown}")
    return values


def _resolved_batch_size(raw_batch_size: int | None) -> int:
    if raw_batch_size is not None:
        return raw_batch_size
    return get_field_kernel_tuning().coil_chunk_size


def _make_points(npoints: int) -> jax.Array:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npoints, endpoint=False)
    return jnp.stack(
        (
            0.35 * jnp.cos(theta),
            0.35 * jnp.sin(theta),
            0.12 * jnp.sin(2.0 * theta),
        ),
        axis=1,
    )


def _make_coil_set_spec(ncoils: int, nquad: int) -> GroupedCoilSetSpec:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, nquad, endpoint=False)
    gammas = []
    gammadashs = []
    for coil_index in range(ncoils):
        phase = 2.0 * jnp.pi * coil_index / ncoils
        radius = 0.82 + 0.01 * coil_index
        z_scale = 0.07 + 0.002 * coil_index
        angle = theta + phase
        gamma = jnp.stack(
            (
                radius * jnp.cos(angle),
                radius * jnp.sin(angle),
                z_scale * jnp.sin(2.0 * angle),
            ),
            axis=1,
        )
        gammadash = jnp.stack(
            (
                -radius * jnp.sin(angle),
                radius * jnp.cos(angle),
                2.0 * z_scale * jnp.cos(2.0 * angle),
            ),
            axis=1,
        )
        gammas.append(gamma)
        gammadashs.append(gammadash)
    return GroupedCoilSetSpec(
        groups=(
            CoilGroupSpec(
                gammas=jnp.stack(gammas),
                gammadashs=jnp.stack(gammadashs),
                currents=jnp.ones((ncoils,), dtype=jnp.float64),
                coil_indices=tuple(range(ncoils)),
            ),
        )
    )


def _serial_per_coil_unit_field(points, coil_set_spec, kernel):
    ncoils = sum(len(group.coil_indices) for group in coil_set_spec.groups)
    result_by_index = {}
    for group in coil_set_spec.groups:
        unit_current = jnp.ones((1,), dtype=group.currents.dtype)
        for position, coil_index in enumerate(group.coil_indices):
            result_by_index[int(coil_index)] = kernel(
                points,
                group.gammas[position][jnp.newaxis, ...],
                group.gammadashs[position][jnp.newaxis, ...],
                unit_current,
            )
    return tuple(result_by_index[index] for index in range(ncoils))


def _unbounded_vmap_per_coil_unit_field(points, coil_set_spec, kernel):
    ncoils = sum(len(group.coil_indices) for group in coil_set_spec.groups)
    result_by_index = {}
    for group in coil_set_spec.groups:
        unit_current = jnp.ones((1,), dtype=group.currents.dtype)

        def evaluate_single(gamma, gammadash):
            return kernel(
                points,
                gamma[jnp.newaxis, ...],
                gammadash[jnp.newaxis, ...],
                unit_current,
            )

        group_results = jax.vmap(evaluate_single)(group.gammas, group.gammadashs)
        for position, coil_index in enumerate(group.coil_indices):
            result_by_index[int(coil_index)] = group_results[position]
    return tuple(result_by_index[index] for index in range(ncoils))


def _bounded_lax_map_per_coil_unit_field(points, coil_set_spec, kernel, batch_size):
    return tuple(
        _per_coil_unit_field_with_batch_size(
            points,
            coil_set_spec,
            kernel,
            batch_size=batch_size,
        )
    )


def _current_per_coil_unit_field(points, coil_set_spec):
    return tuple(_per_coil_unit_field(points, coil_set_spec, biot_savart_B))


def _block_ready(value):
    return jax.block_until_ready(value)


def _time_call(fn, points, coil_set_spec, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        _block_ready(fn(points, coil_set_spec))
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        _block_ready(fn(points, coil_set_spec))
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _mode_fn(mapping_mode: str, kernel_name: str, batch_size: int):
    kernel = _KERNELS[kernel_name]
    if mapping_mode == "unbounded_vmap":
        return lambda points, spec: _unbounded_vmap_per_coil_unit_field(
            points,
            spec,
            kernel,
        )
    if mapping_mode == "bounded_lax_map":
        return lambda points, spec: _bounded_lax_map_per_coil_unit_field(
            points,
            spec,
            kernel,
            batch_size,
        )
    raise ValueError(f"unknown mapping mode: {mapping_mode}")


def _summarize_output(value) -> dict[str, float | int]:
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(value)]
    total_bytes = int(sum(leaf.nbytes for leaf in leaves))
    total_size = int(sum(leaf.size for leaf in leaves))
    max_abs = 0.0
    for leaf in leaves:
        if leaf.size:
            max_abs = max(max_abs, float(np.max(np.abs(leaf))))
    return {
        "leaf_count": len(leaves),
        "element_count": total_size,
        "output_bytes": total_bytes,
        "max_abs_output": max_abs,
    }


def _memory_analysis_summary(compiled) -> dict[str, int]:
    analysis = compiled.memory_analysis()
    return {
        "generated_code_size_in_bytes": int(analysis.generated_code_size_in_bytes),
        "argument_size_in_bytes": int(analysis.argument_size_in_bytes),
        "output_size_in_bytes": int(analysis.output_size_in_bytes),
        "alias_size_in_bytes": int(analysis.alias_size_in_bytes),
        "temp_size_in_bytes": int(analysis.temp_size_in_bytes),
        "host_generated_code_size_in_bytes": int(
            analysis.host_generated_code_size_in_bytes
        ),
        "host_argument_size_in_bytes": int(analysis.host_argument_size_in_bytes),
        "host_output_size_in_bytes": int(analysis.host_output_size_in_bytes),
        "host_alias_size_in_bytes": int(analysis.host_alias_size_in_bytes),
        "host_temp_size_in_bytes": int(analysis.host_temp_size_in_bytes),
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


def _compile_and_time_child(args: argparse.Namespace) -> dict[str, object]:
    if args.child_mapping_mode is None:
        raise ValueError("--child-mapping-mode is required for child probes.")
    if args.child_kernel is None:
        raise ValueError("--child-kernel is required for child probes.")
    if args.child_ncoils is None:
        raise ValueError("--child-ncoils is required for child probes.")

    batch_size = _resolved_batch_size(args.batch_size)
    points = _make_points(args.npoints)
    coil_set_spec = _make_coil_set_spec(args.child_ncoils, args.nquad)
    fn = jax.jit(_mode_fn(args.child_mapping_mode, args.child_kernel, batch_size))

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
            compiled = fn.lower(points, coil_set_spec).compile()
            compile_s = time.perf_counter() - compile_started_at
            execute_started_at = time.perf_counter()
            output = _block_ready(compiled(points, coil_set_spec))
            first_execute_s = time.perf_counter() - execute_started_at
            for _ in range(args.warmup):
                _block_ready(compiled(points, coil_set_spec))
            timings = []
            for _ in range(args.repeat):
                repeat_started_at = time.perf_counter()
                _block_ready(compiled(points, coil_set_spec))
                timings.append(time.perf_counter() - repeat_started_at)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    peak_rss_mb = _peak_rss_mb()
    memory_analysis = _memory_analysis_summary(compiled)
    return {
        "mapping_mode": args.child_mapping_mode,
        "kernel": args.child_kernel,
        "ncoils": int(args.child_ncoils),
        "npoints": int(args.npoints),
        "nquad": int(args.nquad),
        "batch_size": int(batch_size),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
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
        "compiled_memory_analysis": memory_analysis,
        "output_summary": _summarize_output(output),
    }


def _assert_outputs_match(current_fn, serial_fn, points, coil_set_spec) -> None:
    current = current_fn(points, coil_set_spec)
    serial = serial_fn(points, coil_set_spec)
    for current_leaf, serial_leaf in zip(
        jax.tree.leaves(current),
        jax.tree.leaves(serial),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(current_leaf),
            np.asarray(serial_leaf),
            rtol=1e-12,
            atol=1e-12,
        )


def _log_slope(rows: list[dict[str, float]], key: str) -> float:
    x = np.log(np.asarray([row["ncoils"] for row in rows], dtype=float))
    y = np.log(np.asarray([row[key] for row in rows], dtype=float))
    return float(np.polyfit(x, y, deg=1)[0])


def _run_child_process(
    args: argparse.Namespace,
    *,
    mapping_mode: str,
    kernel_name: str,
    ncoils: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mapping-mode",
        mapping_mode,
        "--child-kernel",
        kernel_name,
        "--child-ncoils",
        str(ncoils),
        "--npoints",
        str(args.npoints),
        "--nquad",
        str(args.nquad),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
    ]
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "per-coil child probe failed with return code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n"
            f"{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _memory_analysis_delta(
    unbounded: dict[str, object],
    bounded: dict[str, object],
) -> dict[str, int]:
    unbounded_memory = unbounded["compiled_memory_analysis"]
    bounded_memory = bounded["compiled_memory_analysis"]
    return {
        key: int(bounded_memory[key]) - int(unbounded_memory[key])
        for key in unbounded_memory
    }


def _batching_comparison(args: argparse.Namespace) -> dict[str, object]:
    ncoils = max(_parse_ncoils(args.ncoils))
    batch_size = _resolved_batch_size(args.batch_size)
    points = _make_points(args.npoints)
    coil_set_spec = _make_coil_set_spec(ncoils, args.nquad)
    cases = []
    for kernel_name in _parse_kernels(args.kernels):
        unbounded_output = _unbounded_vmap_per_coil_unit_field(
            points,
            coil_set_spec,
            _KERNELS[kernel_name],
        )
        bounded_output = _bounded_lax_map_per_coil_unit_field(
            points,
            coil_set_spec,
            _KERNELS[kernel_name],
            batch_size,
        )
        unbounded = _run_child_process(
            args,
            mapping_mode="unbounded_vmap",
            kernel_name=kernel_name,
            ncoils=ncoils,
        )
        bounded = _run_child_process(
            args,
            mapping_mode="bounded_lax_map",
            kernel_name=kernel_name,
            ncoils=ncoils,
        )
        cases.append(
            {
                "kernel": kernel_name,
                "unbounded_vmap": unbounded,
                "bounded_lax_map": bounded,
                "delta_bounded_minus_unbounded": {
                    "compile_log_count": int(bounded["compile_log_count"])
                    - int(unbounded["compile_log_count"]),
                    "first_call_s": float(bounded["first_call_s"])
                    - float(unbounded["first_call_s"]),
                    "post_compile_median_s": float(bounded["post_compile_median_s"])
                    - float(unbounded["post_compile_median_s"]),
                    "peak_rss_mb": float(bounded["peak_rss_mb"])
                    - float(unbounded["peak_rss_mb"]),
                    "peak_rss_delta_mb": float(bounded["peak_rss_delta_mb"])
                    - float(unbounded["peak_rss_delta_mb"]),
                    "compiled_memory_analysis": _memory_analysis_delta(
                        unbounded,
                        bounded,
                    ),
                },
                "output_delta": _output_delta(unbounded_output, bounded_output),
            }
        )
    return {
        "measurement": "per-coil unit-field unbounded-vmap vs bounded-lax-map",
        "ncoils": int(ncoils),
        "npoints": int(args.npoints),
        "nquad": int(args.nquad),
        "batch_size": int(batch_size),
        "kernels": list(_parse_kernels(args.kernels)),
        "cases": cases,
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    current_fn = jax.jit(_current_per_coil_unit_field)
    single_coil_kernel = jax.jit(biot_savart_B)
    ncoils_values = _parse_ncoils(args.ncoils)

    def serial_fn(points, spec):
        return _serial_per_coil_unit_field(
            points,
            spec,
            single_coil_kernel,
        )

    points = _make_points(args.npoints)
    rows = []
    for ncoils in ncoils_values:
        coil_set_spec = _make_coil_set_spec(ncoils, args.nquad)
        _assert_outputs_match(current_fn, serial_fn, points, coil_set_spec)
        current_s = _time_call(
            current_fn,
            points,
            coil_set_spec,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        serial_s = _time_call(
            serial_fn,
            points,
            coil_set_spec,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        rows.append(
            {
                "ncoils": ncoils,
                "current_median_s": current_s,
                "serial_median_s": serial_s,
                "speedup": serial_s / current_s,
            }
        )
    summary = {
        "rows": rows,
        "current_loglog_slope": _log_slope(rows, "current_median_s"),
        "serial_loglog_slope": _log_slope(rows, "serial_median_s"),
    }
    if args.compare_vmap_batching:
        summary["batched_vs_unbounded"] = _batching_comparison(args)
    has_slope_evidence = len(ncoils_values) >= 3
    slope_regressed = (
        has_slope_evidence
        and summary["current_loglog_slope"] >= summary["serial_loglog_slope"]
    )
    if args.fail_on_regression and (slope_regressed or rows[-1]["speedup"] <= 1.0):
        raise RuntimeError(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    if args.child_mapping_mode is not None:
        result = _compile_and_time_child(args)
    else:
        result = run_probe(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
