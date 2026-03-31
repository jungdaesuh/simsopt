"""Memory-scaling benchmark for the pure JAX Biot-Savart kernels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from benchmarks.validation_ladder_common import (
    bootstrap_local_simsopt,
    build_provenance,
    print_provenance,
    write_json,
)

bootstrap_local_simsopt()

from simsopt import backend as simsopt_backend
from simsopt.jax_core import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
)


@dataclass(frozen=True)
class KernelScalingCase:
    label: str
    ncoils: int
    nquad: int
    npoints: int


_SCALING_CASES = (
    KernelScalingCase("small", ncoils=8, nquad=64, npoints=64),
    KernelScalingCase("medium", ncoils=16, nquad=128, npoints=256),
    KernelScalingCase("large", ncoils=32, nquad=256, npoints=512),
)
_KERNEL_BENCHMARKS = (
    ("B", biot_savart_B),
    ("A", biot_savart_A),
    ("dB_by_dX", biot_savart_dB_by_dX),
    ("dA_by_dX", biot_savart_dA_by_dX),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=simsopt_backend.VALID_BACKEND_MODES,
        default="jax_cpu_parity",
        help="Backend mode used to evaluate the kernel.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def _make_fixture(case: KernelScalingCase, *, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(case.npoints, 3))
    points[:, 0] -= 2.0
    gammas = rng.normal(size=(case.ncoils, case.nquad, 3))
    gammas[:, :, 0] += 1.5
    gammadashs = rng.normal(size=(case.ncoils, case.nquad, 3))
    currents = rng.normal(loc=1.0e5, scale=2.0e4, size=(case.ncoils,))
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        jnp.asarray(gammadashs, dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _estimate_input_bytes(points, gammas, gammadashs, currents) -> int:
    return int(points.nbytes + gammas.nbytes + gammadashs.nbytes + currents.nbytes)


def _measure_kernel(
    fn: Callable,
    *args,
    warmup: int,
    repeat: int,
) -> dict[str, float | int | list[int]]:
    t0 = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        durations.append(time.perf_counter() - t0)

    return {
        "compile_s": float(compile_s),
        "median_ms": float(np.median(durations) * 1e3),
        "mean_ms": float(np.mean(durations) * 1e3),
        "shape": list(np.shape(result)),
    }


def _measure_case_kernels(
    points, gammas, gammadashs, currents, *, warmup: int, repeat: int
):
    return {
        name: _measure_kernel(
            fn,
            points,
            gammas,
            gammadashs,
            currents,
            warmup=warmup,
            repeat=repeat,
        )
        for name, fn in _KERNEL_BENCHMARKS
    }


def build_biotsavart_kernel_scaling_payload(
    *,
    title: str,
    mode: str,
    warmup: int,
    repeat: int,
    seed: int,
    cases: tuple[KernelScalingCase, ...] = _SCALING_CASES,
) -> dict[str, object]:
    tuning = simsopt_backend.get_field_kernel_tuning(mode)
    provenance = build_provenance(
        jax,
        jaxlib,
        title=title,
        extra={
            "lane": "kernel-scaling",
            "backend_mode": mode,
            "chunk_policy": tuning.chunk_policy,
            "coil_chunk_size": tuning.coil_chunk_size,
            "quadrature_block_size": tuning.quadrature_block_size,
            "compile_behavior": "cold+warm",
        },
    )
    case_payloads = []
    for index, case in enumerate(cases):
        points, gammas, gammadashs, currents = _make_fixture(case, seed=seed + index)
        case_payloads.append(
            {
                **asdict(case),
                "input_bytes": _estimate_input_bytes(
                    points,
                    gammas,
                    gammadashs,
                    currents,
                ),
                **_measure_case_kernels(
                    points,
                    gammas,
                    gammadashs,
                    currents,
                    warmup=warmup,
                    repeat=repeat,
                ),
            }
        )
    return {"provenance": provenance, "cases": case_payloads}


def main() -> None:
    args = _parse_args()
    simsopt_backend.set_backend(args.mode)
    payload = build_biotsavart_kernel_scaling_payload(
        title="Biot-Savart kernel scaling benchmark",
        mode=args.mode,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )
    print_provenance(payload["provenance"])
    for case_payload in payload["cases"]:
        print(
            f"{case_payload['label']}: coils={case_payload['ncoils']} "
            f"nquad={case_payload['nquad']} points={case_payload['npoints']} "
            f"input={case_payload['input_bytes']}B"
        )
        for kernel_name in ("B", "A", "dB_by_dX", "dA_by_dX"):
            kernel_payload = case_payload[kernel_name]
            print(
                f"  {kernel_name}: compile={kernel_payload['compile_s']:.3f}s "
                f"median={kernel_payload['median_ms']:.3f}ms "
                f"mean={kernel_payload['mean_ms']:.3f}ms"
            )
    if args.output_json:
        write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
