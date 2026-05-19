"""Real-device timing probe for round-3 multi-GPU sharding contracts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:], default_platform="cuda")
bootstrap_local_simsopt(SRC_ROOT)

import jax
import jaxlib
import jax.numpy as jnp
from jax import lax

from simsopt.backend import get_sharding_tuning
from simsopt.geo import surfaceobjectives_jax as surfaceobjectives_jax_module
from simsopt.jax_core.integral_bdotn import (
    integral_BdotN,
    integral_BdotN_sharding_summary,
    integral_BdotN_surface_sharded,
)
from simsopt.jax_core.sharding import (
    maybe_shard_seed_batch_inputs,
    maybe_shard_surface_quadrature_inputs,
    seed_batch_sharding_config,
    seed_batch_sharding_summary,
    surface_quadrature_sharding_config,
)


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        choices=("integral-bdotn", "seed-batch"),
        required=True,
    )
    parser.add_argument("--nphi", type=int, default=4096)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--seed-dofs", type=int, default=256)
    parser.add_argument("--seed-work", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--gpu-memory-sample-s",
        type=float,
        default=0.25,
        help="Set to 0 to skip nvidia-smi sampling in non-GPU smoke tests.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


class GpuMemorySampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.visible_gpu_indices = _visible_gpu_indices(self.cuda_visible_devices)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._snapshots: list[dict[str, object]] = []
        self._error: BaseException | None = None

    def __enter__(self) -> "GpuMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        if exc_type is None and self._error is not None:
            raise RuntimeError("nvidia-smi sampling failed") from self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except BaseException as exc:  # propagate from __exit__
                self._error = exc
                self._stop.set()
                return
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        per_gpu = []
        for line in result.stdout.splitlines():
            index_text, memory_text = (field.strip() for field in line.split(",", 1))
            per_gpu.append(
                {
                    "index": int(index_text),
                    "memory_used_mib": int(memory_text),
                }
            )
        self._snapshots.append({"time_s": time.time(), "gpus": per_gpu})

    def summary(self) -> dict[str, object]:
        max_per_gpu: dict[int, int] = {}
        baseline_per_gpu: dict[int, int] = {}
        max_total_mib = 0
        for snapshot in self._snapshots:
            total_mib = 0
            for gpu_row in snapshot["gpus"]:
                index = int(gpu_row["index"])
                if (
                    self.visible_gpu_indices is not None
                    and index not in self.visible_gpu_indices
                ):
                    continue
                used_mib = int(gpu_row["memory_used_mib"])
                baseline_per_gpu.setdefault(index, used_mib)
                max_per_gpu[index] = max(max_per_gpu.get(index, 0), used_mib)
                total_mib += used_mib
            max_total_mib = max(max_total_mib, total_mib)
        max_delta_per_gpu = {
            index: max_per_gpu[index] - baseline_per_gpu[index]
            for index in sorted(max_per_gpu)
        }
        return {
            "cuda_visible_devices": self.cuda_visible_devices,
            "visible_gpu_indices": None
            if self.visible_gpu_indices is None
            else list(self.visible_gpu_indices),
            "sample_count": len(self._snapshots),
            "max_per_gpu_mib": dict(sorted(max_per_gpu.items())),
            "max_delta_per_gpu_mib": max_delta_per_gpu,
            "max_total_mib": max_total_mib,
            "max_visible_total_delta_mib": sum(max_delta_per_gpu.values()),
        }


class DisabledGpuMemorySampler:
    def __enter__(self) -> "DisabledGpuMemorySampler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def summary(self) -> dict[str, object]:
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_indices": None,
            "sample_count": 0,
            "max_per_gpu_mib": {},
            "max_delta_per_gpu_mib": {},
            "max_total_mib": None,
            "max_visible_total_delta_mib": None,
        }


def _gpu_memory_sampler(
    interval_s: float,
) -> GpuMemorySampler | DisabledGpuMemorySampler:
    if interval_s <= 0.0:
        return DisabledGpuMemorySampler()
    return GpuMemorySampler(interval_s)


def _visible_gpu_indices(cuda_visible_devices: str | None) -> tuple[int, ...] | None:
    if cuda_visible_devices is None or cuda_visible_devices.strip() == "":
        return None
    return tuple(int(field) for field in cuda_visible_devices.split(","))


def _block_ready(value):
    return jax.block_until_ready(value)


def _time_call(fn, *args, warmup: int, repeat: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        _block_ready(fn(*args))
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        _block_ready(fn(*args))
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), timings


def _make_integral_inputs(nphi: int, ntheta: int) -> tuple[jax.Array, ...]:
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)[:, None]
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)[None, :]
    normal = jnp.stack(
        (
            (1.0 + 0.08 * jnp.cos(theta)) * jnp.cos(phi),
            (1.0 + 0.08 * jnp.cos(theta)) * jnp.sin(phi),
            0.08 * jnp.sin(theta) + jnp.zeros_like(phi),
        ),
        axis=2,
    )
    Bcoil = jnp.stack(
        (
            0.32 + 0.05 * jnp.sin(phi + theta),
            0.27 + 0.04 * jnp.cos(2.0 * phi - theta),
            0.19 + 0.03 * jnp.sin(phi - 2.0 * theta),
        ),
        axis=2,
    )
    target = 0.01 * jnp.sin(2.0 * phi + theta)
    return Bcoil, target, normal


def _collective_metadata(lowered_text: str) -> dict[str, object]:
    lowered = lowered_text.lower()
    all_reduce_count = lowered.count("all_reduce") + lowered.count("all-reduce")
    return {
        "all_reduce_count": all_reduce_count,
        "contains_collective": all_reduce_count > 0,
        "hlo_collective_bytes": None,
        "hlo_collective_bytes_note": (
            "StableHLO text exposes collective ops but not a portable byte counter; "
            "payload size is derived from the scalar all-reduce contract."
        ),
    }


def run_integral_bdotn_probe(args: argparse.Namespace) -> dict[str, object]:
    Bcoil, target, normal = _make_integral_inputs(args.nphi, args.ntheta)
    reference_fn = jax.jit(integral_BdotN, static_argnames=("definition",))
    sharded_fn = jax.jit(
        integral_BdotN_surface_sharded,
        static_argnames=("definition",),
    )
    definition = "quadratic flux"
    reference_value = reference_fn(Bcoil, target, normal, definition)
    config = surface_quadrature_sharding_config(Bcoil)
    if config is None:
        timed_Bcoil, timed_target, timed_normal = Bcoil, target, normal
    else:
        timed_Bcoil, timed_target, timed_normal = maybe_shard_surface_quadrature_inputs(
            Bcoil,
            target,
            normal,
            config=config,
        )
    sharded_value = sharded_fn(timed_Bcoil, timed_target, timed_normal, definition)
    np.testing.assert_allclose(
        np.asarray(sharded_value),
        np.asarray(reference_value),
        rtol=1e-12,
        atol=1e-12,
    )
    sharded_median_s, sharded_timings_s = _time_call(
        sharded_fn,
        timed_Bcoil,
        timed_target,
        timed_normal,
        definition,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    reference_median_s, reference_timings_s = _time_call(
        reference_fn,
        Bcoil,
        target,
        normal,
        definition,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    lowered_text = sharded_fn.lower(
        timed_Bcoil,
        timed_target,
        timed_normal,
        definition,
    ).as_text()
    return {
        "probe": "integral-bdotn",
        "nphi": args.nphi,
        "ntheta": args.ntheta,
        "value": float(np.asarray(sharded_value)),
        "reference_value": float(np.asarray(reference_value)),
        "abs_error": float(
            abs(np.asarray(sharded_value) - np.asarray(reference_value))
        ),
        "sharded_median_s": sharded_median_s,
        "sharded_timings_s": sharded_timings_s,
        "reference_median_s": reference_median_s,
        "reference_timings_s": reference_timings_s,
        "sharding_summary": integral_BdotN_sharding_summary(
            timed_Bcoil,
            timed_target,
            timed_normal,
            definition,
        ),
        "collectives": _collective_metadata(lowered_text),
        "input_placement": "pre_sharded" if config is not None else "single_device",
    }


def _seed_scalar_objective(coil_dofs: jax.Array, work: int) -> jax.Array:
    def body(_, value):
        return jnp.sin(value) + 0.25 * jnp.cos(0.5 * value)

    transformed = lax.fori_loop(0, work, body, coil_dofs)
    return jnp.vdot(transformed, transformed) / transformed.size


def run_seed_batch_probe(args: argparse.Namespace) -> dict[str, object]:
    compiled_value_and_grad_for = jax.jit(
        jax.value_and_grad(
            lambda coil_dofs: _seed_scalar_objective(coil_dofs, args.seed_work)
        )
    )
    batched_value_and_grad = (
        surfaceobjectives_jax_module._make_traceable_batched_value_and_grad_pipeline(
            compiled_value_and_grad_for
        )
    )
    coil_dofs_batch = jnp.linspace(
        -1.0,
        1.0,
        args.seed_count * args.seed_dofs,
        dtype=jnp.float64,
    ).reshape(args.seed_count, args.seed_dofs)
    reference_values, reference_grads = jax.vmap(compiled_value_and_grad_for)(
        coil_dofs_batch
    )
    config = seed_batch_sharding_config(coil_dofs_batch)
    if config is None:
        timed_coil_dofs_batch = coil_dofs_batch
    else:
        (timed_coil_dofs_batch,) = maybe_shard_seed_batch_inputs(
            coil_dofs_batch,
            config=config,
        )
    values, grads = batched_value_and_grad(timed_coil_dofs_batch)
    np.testing.assert_allclose(
        np.asarray(values),
        np.asarray(reference_values),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads),
        np.asarray(reference_grads),
        rtol=1e-12,
        atol=1e-12,
    )
    sharded_median_s, sharded_timings_s = _time_call(
        batched_value_and_grad,
        timed_coil_dofs_batch,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    reference_fn = jax.jit(lambda batch: jax.vmap(compiled_value_and_grad_for)(batch))
    reference_median_s, reference_timings_s = _time_call(
        reference_fn,
        coil_dofs_batch,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    lowered_text = batched_value_and_grad.lower(timed_coil_dofs_batch).as_text()
    return {
        "probe": "seed-batch",
        "seed_count": args.seed_count,
        "seed_dofs": args.seed_dofs,
        "seed_work": args.seed_work,
        "value_sum": float(np.asarray(jnp.sum(values))),
        "reference_value_sum": float(np.asarray(jnp.sum(reference_values))),
        "max_grad_abs_error": float(
            np.max(np.abs(np.asarray(grads - reference_grads)))
        ),
        "sharded_median_s": sharded_median_s,
        "sharded_timings_s": sharded_timings_s,
        "reference_median_s": reference_median_s,
        "reference_timings_s": reference_timings_s,
        "sharding_summary": seed_batch_sharding_summary(values, config=config),
        "input_sharding_summary": seed_batch_sharding_summary(
            timed_coil_dofs_batch, config=config
        ),
        "collectives": _collective_metadata(lowered_text),
        "input_placement": "pre_sharded" if config is not None else "single_device",
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    with _gpu_memory_sampler(args.gpu_memory_sample_s) as sampler:
        if args.probe == "integral-bdotn":
            result = run_integral_bdotn_probe(args)
        else:
            result = run_seed_batch_probe(args)
    result["runtime"] = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "local_device_count": jax.local_device_count(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "sharding_tuning": dataclasses.asdict(get_sharding_tuning()),
    }
    result["gpu_memory"] = sampler.summary()
    return result


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
