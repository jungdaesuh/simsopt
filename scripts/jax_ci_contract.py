"""GPU CI contract probes for reduction order, drift logging, and reproducibility."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_compilation_cache_policy,
    apply_requested_platform,
    build_provenance,
    describe_compile_behavior,
    preparse_platform,
    print_provenance,
    relative_error,
    require_x64_runtime,
    write_json,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    ci_reproducibility_contract,
    ratchet_rel_tol,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_compilation_cache_policy()

import jax  # noqa: E402
import jaxlib  # noqa: E402
import jax.numpy as jnp  # noqa: E402


jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="JAX CI contract probes")


def _ordered_reduction_values(sample_size: int) -> np.ndarray:
    return np.logspace(12.0, -12.0, sample_size, dtype=np.float64)


def _as_positive_ulp_distance(actual: float, reference: float) -> int:
    actual_bits = np.asarray(actual, dtype=np.float64).view(np.uint64)
    reference_bits = np.asarray(reference, dtype=np.float64).view(np.uint64)
    return int(abs(int(actual_bits) - int(reference_bits)))


def _sum_on_backend(values: np.ndarray, *, backend: str) -> float:
    sum_fn = jax.jit(lambda x: jnp.sum(x), backend=backend)
    return float(sum_fn(jnp.asarray(values, dtype=jnp.float64)))


def _lane_label(requested_platform: str) -> str:
    return os.environ.get(
        "SIMSOPT_BACKEND_MODE",
        "jax_gpu_parity" if requested_platform == "auto" else f"jax_{requested_platform}_parity",
    )


def probe_reduction_order(
    sample_size: int,
    *,
    target_backend: str,
    max_ulp: int,
) -> dict[str, Any]:
    values = _ordered_reduction_values(sample_size)
    cpu_sum = _sum_on_backend(values, backend="cpu")
    backend_sum = _sum_on_backend(values, backend=target_backend)
    rel_err = relative_error(backend_sum, cpu_sum)
    ulp_distance = _as_positive_ulp_distance(backend_sum, cpu_sum)
    return {
        "sample_size": int(sample_size),
        "cpu_sum": cpu_sum,
        "backend_sum": backend_sum,
        "rel_err": rel_err,
        "ulp_distance": ulp_distance,
        "passed": ulp_distance <= int(max_ulp),
    }


def _reproducibility_output(
    *,
    backend: str,
    seed: int,
    sample_size: int,
) -> np.ndarray:
    key = jax.random.PRNGKey(seed)
    values = jax.random.normal(key, (sample_size,), dtype=jnp.float64)
    probe_fn = jax.jit(
        lambda x: jnp.array(
            [
                jnp.sum(x),
                jnp.sum(x * x),
                jnp.max(x),
            ],
            dtype=jnp.float64,
        ),
        backend=backend,
    )
    return np.asarray(probe_fn(values), dtype=np.float64)


def probe_same_device_bitwise_reproducibility(
    *,
    backend: str,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    first = _reproducibility_output(
        backend=backend,
        seed=seed,
        sample_size=sample_size,
    )
    second = _reproducibility_output(
        backend=backend,
        seed=seed,
        sample_size=sample_size,
    )
    first_bits = first.view(np.uint64)
    second_bits = second.view(np.uint64)
    bitwise_equal = bool(np.array_equal(first_bits, second_bits))
    rel_err = (
        relative_error(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
        if first.size
        else 0.0
    )
    return {
        "seed": int(seed),
        "sample_size": int(sample_size),
        "first": first.tolist(),
        "second": second.tolist(),
        "bitwise_equal": bitwise_equal,
        "rel_err": rel_err,
        "passed": bitwise_equal,
    }


def build_ci_contract_payload(*, requested_platform: str) -> dict[str, Any]:
    contract = ci_reproducibility_contract()
    target_backend = str(jax.default_backend())
    max_ulp = int(contract["gpu_reduction_order_max_ulp"])
    sample_size = int(contract["gpu_reduction_order_sample_size"])
    seed = int(contract["gpu_reproducibility_seed"])
    reproducibility_sample_size = int(contract["gpu_reproducibility_sample_size"])
    current_rel_tol = float(contract["gpu_reduction_order_rel_tol"])
    ratchet_factor = float(contract["tolerance_ratchet_factor"])
    reduction_order = probe_reduction_order(
        sample_size,
        target_backend=target_backend,
        max_ulp=max_ulp,
    )
    bitwise_repro = probe_same_device_bitwise_reproducibility(
        backend=target_backend,
        seed=seed,
        sample_size=reproducibility_sample_size,
    )
    ratcheted_rel_tol = ratchet_rel_tol(
        current_rel_tol,
        float(reduction_order["rel_err"]),
        factor=ratchet_factor,
    )
    passed = bool(reduction_order["passed"] and bitwise_repro["passed"])
    provenance_extra = {
        "lane": _lane_label(requested_platform),
        "fixture": "gpu-ci-contract",
        "platform_request": requested_platform,
        "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
        "reduction_order_max_ulp": max_ulp,
        "reduction_order_sample_size": sample_size,
        "reduction_order_rel_tol": current_rel_tol,
        "reduction_order_rel_err": reduction_order["rel_err"],
        "ratcheted_rel_tol": ratcheted_rel_tol,
        "bitwise_repro_seed": seed,
        "bitwise_repro_sample_size": reproducibility_sample_size,
    }
    return {
        "provenance": build_provenance(
            jax,
            jaxlib,
            title="JAX CI reproducibility contract",
            extra=provenance_extra,
        ),
        "policy": {
            "gpu_reduction_order_max_ulp": max_ulp,
            "gpu_reduction_order_sample_size": sample_size,
            "gpu_reduction_order_rel_tol": current_rel_tol,
            "gpu_reproducibility_seed": seed,
            "gpu_reproducibility_sample_size": reproducibility_sample_size,
            "tolerance_ratchet_factor": ratchet_factor,
        },
        "reduction_order": reduction_order,
        "same_device_bitwise_reproducibility": bitwise_repro,
        "tolerance_drift": {
            "achieved_rel_err": float(reduction_order["rel_err"]),
            "current_rel_tol": current_rel_tol,
            "ratchet_factor": ratchet_factor,
            "ratcheted_rel_tol": ratcheted_rel_tol,
        },
        "passed": passed,
    }


def _render_report(payload: dict[str, Any]) -> str:
    provenance = payload["provenance"]
    reduction_order = payload["reduction_order"]
    reproducibility = payload["same_device_bitwise_reproducibility"]
    tolerance = payload["tolerance_drift"]
    return f"""# JAX CI reproducibility contract

## Run Identity

- repo sha: `{provenance["repo_sha"]}`
- backend: `{provenance["backend"]}`
- devices: `{provenance["devices"]}`
- lane: `{provenance.get("lane", "n/a")}`
- platform request: `{provenance.get("platform_request", "n/a")}`

## Reduction Order

- sample size: `{reduction_order["sample_size"]}`
- CPU sum: `{reduction_order["cpu_sum"]:.16e}`
- backend sum: `{reduction_order["backend_sum"]:.16e}`
- rel err: `{reduction_order["rel_err"]:.3e}`
- ULP distance: `{reduction_order["ulp_distance"]}`
- gate: `<= {payload["policy"]["gpu_reduction_order_max_ulp"]} ULP`

## Same-Device Bitwise Reproducibility

- seed: `{reproducibility["seed"]}`
- sample size: `{reproducibility["sample_size"]}`
- bitwise equal: `{str(bool(reproducibility["bitwise_equal"])).lower()}`

## Tolerance Drift

- achieved rel err: `{tolerance["achieved_rel_err"]:.3e}`
- current rel tol: `{tolerance["current_rel_tol"]:.3e}`
- ratchet factor: `{tolerance["ratchet_factor"]:.1f}x`
- ratcheted rel tol: `{tolerance["ratcheted_rel_tol"]:.3e}`

## Result

- passed: `{str(bool(payload["passed"])).lower()}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the JAX CI reproducibility contract probes."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the structured probe payload.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path to write a markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_ci_contract_payload(requested_platform=args.platform)
    print_provenance(payload["provenance"])
    reduction_order = payload["reduction_order"]
    reproducibility = payload["same_device_bitwise_reproducibility"]
    tolerance = payload["tolerance_drift"]
    print(
        "Reduction-order probe: "
        f"ulp_distance={reduction_order['ulp_distance']}, "
        f"rel_err={reduction_order['rel_err']:.3e}, "
        f"gate={payload['policy']['gpu_reduction_order_max_ulp']} ULP"
    )
    print(
        "Same-device bitwise reproducibility: "
        f"bitwise_equal={str(bool(reproducibility['bitwise_equal'])).lower()}"
    )
    print(
        "Tolerance drift: "
        f"achieved_rel_err={tolerance['achieved_rel_err']:.3e}, "
        f"current_rel_tol={tolerance['current_rel_tol']:.3e}, "
        f"ratcheted_rel_tol={tolerance['ratcheted_rel_tol']:.3e}"
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)
    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
