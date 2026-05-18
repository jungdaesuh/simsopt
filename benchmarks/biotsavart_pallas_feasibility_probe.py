"""Profile current Biot-Savart XLA kernels before considering Pallas/Triton."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repo_bootstrap import configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from benchmarks.validation_ladder_common import (
    bootstrap_local_simsopt,
    build_provenance,
    write_json,
)

bootstrap_local_simsopt()

from simsopt.jax_core.biotsavart import biot_savart_B, biot_savart_B_vjp


@dataclass(frozen=True)
class BiotSavartFeasibilityShape:
    ncoils: int
    nquad: int
    npoints: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before profiling.",
    )
    parser.add_argument("--ncoils", type=int, default=8)
    parser.add_argument("--nquad", type=int, default=64)
    parser.add_argument("--npoints", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def _make_fixture(shape: BiotSavartFeasibilityShape, *, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(shape.npoints, 3))
    points[:, 0] += 1.5
    gammas = rng.normal(size=(shape.ncoils, shape.nquad, 3))
    gammas[:, :, 0] += 1.0
    gammadashs = rng.normal(size=(shape.ncoils, shape.nquad, 3))
    currents = rng.normal(loc=8.0e4, scale=1.0e4, size=(shape.ncoils,))
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        jnp.asarray(gammadashs, dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _memory_analysis_payload(compiled) -> dict[str, int]:
    stats = compiled.memory_analysis()
    return {
        "argument_size_in_bytes": int(stats.argument_size_in_bytes),
        "output_size_in_bytes": int(stats.output_size_in_bytes),
        "alias_size_in_bytes": int(stats.alias_size_in_bytes),
        "temp_size_in_bytes": int(stats.temp_size_in_bytes),
        "generated_code_size_in_bytes": int(stats.generated_code_size_in_bytes),
    }


def _time_compiled(compiled, *args, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    timings = []
    for _ in range(repeat):
        started = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        timings.append(time.perf_counter() - started)
    return {
        "min_s": float(min(timings)),
        "mean_s": float(sum(timings) / len(timings)),
        "max_s": float(max(timings)),
    }


def _hlo_counts(text: str) -> dict[str, int]:
    return {
        "all_reduce": text.count("all-reduce"),
        "broadcast": text.count("broadcast"),
        "divide": text.count("divide"),
        "fusion": text.count("fusion"),
        "multiply": text.count("multiply"),
        "reduce": text.count("reduce"),
        "rsqrt": text.count("rsqrt"),
        "subtract": text.count("subtract"),
        "transpose": text.count("transpose"),
    }


def _estimated_intermediate_bytes(
    shape: BiotSavartFeasibilityShape,
) -> list[dict[str, int | str]]:
    scalar_grid = shape.npoints * shape.ncoils * shape.nquad
    vector_grid = scalar_grid * 3
    float64_bytes = 8
    estimates = [
        ("point_minus_gamma", vector_grid * float64_bytes),
        ("cross_gammadash_residual", vector_grid * float64_bytes),
        ("weighted_integrand", vector_grid * float64_bytes),
        ("squared_distance", scalar_grid * float64_bytes),
        ("inverse_radius_cubed", scalar_grid * float64_bytes),
    ]
    return [
        {"name": name, "estimated_bytes": int(nbytes)}
        for name, nbytes in sorted(estimates, key=lambda item: item[1], reverse=True)
    ]


def build_biotsavart_pallas_feasibility_payload(
    *,
    shape: BiotSavartFeasibilityShape,
    warmup: int,
    repeat: int,
    seed: int,
) -> dict[str, object]:
    points, gammas, gammadashs, currents = _make_fixture(shape, seed=seed)
    cotangent = jnp.ones((shape.npoints, 3), dtype=jnp.float64)

    value_compiled = jax.jit(biot_savart_B).lower(
        points, gammas, gammadashs, currents
    ).compile()
    vjp_compiled = jax.jit(
        lambda points_arg, cotangent_arg, gammas_arg, gammadashs_arg, currents_arg: (
            biot_savart_B_vjp(
                points_arg,
                cotangent_arg,
                gammas_arg,
                gammadashs_arg,
                currents_arg,
            )
        )
    ).lower(points, cotangent, gammas, gammadashs, currents).compile()

    value_hlo = value_compiled.as_text()
    vjp_hlo = vjp_compiled.as_text()
    has_cuda_device = any(device.platform == "gpu" for device in jax.devices())
    decision_reason = (
        "CUDA profile data is present only when a GPU backend is active; this "
        "probe records current XLA behavior and keeps custom-kernel work out "
        "of the product path until CUDA value/VJP memory data and AD parity "
        "justify the maintenance cost."
    )
    if not has_cuda_device:
        decision_reason = (
            "Local run is CPU-only, so it cannot establish a CUDA HBM "
            "bottleneck. Current XLA CPU value/VJP timing and memory analysis "
            "were recorded; no Pallas/Triton product rewrite is approved."
        )

    return {
        "provenance": build_provenance(
            jax,
            jaxlib,
            title="Biot-Savart Pallas/Triton feasibility probe",
            extra={
                "shape": asdict(shape),
                "warmup": int(warmup),
                "repeat": int(repeat),
                "seed": int(seed),
            },
        ),
        "backend": {
            "default_backend": jax.default_backend(),
            "devices": [
                {"platform": device.platform, "kind": device.device_kind}
                for device in jax.devices()
            ],
        },
        "current_xla": {
            "value": {
                "timing": _time_compiled(
                    value_compiled,
                    points,
                    gammas,
                    gammadashs,
                    currents,
                    warmup=warmup,
                    repeat=repeat,
                ),
                "memory_analysis": _memory_analysis_payload(value_compiled),
                "optimized_hlo_counts": _hlo_counts(value_hlo),
                "optimized_hlo_line_count": len(value_hlo.splitlines()),
            },
            "vjp": {
                "timing": _time_compiled(
                    vjp_compiled,
                    points,
                    cotangent,
                    gammas,
                    gammadashs,
                    currents,
                    warmup=warmup,
                    repeat=repeat,
                ),
                "memory_analysis": _memory_analysis_payload(vjp_compiled),
                "optimized_hlo_counts": _hlo_counts(vjp_hlo),
                "optimized_hlo_line_count": len(vjp_hlo.splitlines()),
            },
        },
        "estimated_dominant_intermediates": _estimated_intermediate_bytes(shape),
        "decision": {
            "pallas_triton_product_rewrite": "no",
            "custom_kernel_prototype_status": "not_started",
            "reason": decision_reason,
        },
    }


def main() -> None:
    args = _parse_args()
    shape = BiotSavartFeasibilityShape(
        ncoils=int(args.ncoils),
        nquad=int(args.nquad),
        npoints=int(args.npoints),
    )
    payload = build_biotsavart_pallas_feasibility_payload(
        shape=shape,
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        seed=int(args.seed),
    )
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
