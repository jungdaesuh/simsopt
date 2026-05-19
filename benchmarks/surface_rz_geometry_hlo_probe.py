"""HLO and timing probe for fused SurfaceRZFourier geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Callable

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

from simsopt.jax_core.surface_rzfourier import (
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_geometry_from_spec,
    surface_rz_fourier_gammadash1_from_spec,
    surface_rz_fourier_gammadash2_from_spec,
    surface_rz_fourier_spec_from_dofs,
)


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and time current scalar-composed versus fused RZ surface "
            "geometry evaluators."
        )
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--nphi", type=int, default=65)
    parser.add_argument("--ntheta", type=int, default=66)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--non-stellsym", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def _dof_count(*, mpol: int, ntor: int, stellsym: bool) -> int:
    include_count = (ntor + 1) + mpol * (2 * ntor + 1)
    exclude_count = ntor + mpol * (2 * ntor + 1)
    if stellsym:
        return include_count + exclude_count
    return 2 * include_count + 2 * exclude_count


def _probe_dofs(args: argparse.Namespace) -> jax.Array:
    stellsym = not args.non_stellsym
    rng = np.random.default_rng(args.seed)
    dofs = rng.normal(
        scale=0.02,
        size=_dof_count(mpol=args.mpol, ntor=args.ntor, stellsym=stellsym),
    )
    dofs[0] = 1.2
    if args.mpol >= 1:
        dofs[args.ntor + 1] += 0.15
    return jnp.asarray(dofs, dtype=jnp.float64)


def _surface_spec_from_dofs(args: argparse.Namespace, dofs: jax.Array):
    return surface_rz_fourier_spec_from_dofs(
        dofs,
        quadpoints_phi=jnp.asarray(
            np.linspace(0.0, 1.0 / args.nfp, args.nphi, endpoint=False),
            dtype=jnp.float64,
        ),
        quadpoints_theta=jnp.asarray(
            np.linspace(0.0, 1.0, args.ntheta, endpoint=False),
            dtype=jnp.float64,
        ),
        mpol=args.mpol,
        ntor=args.ntor,
        nfp=args.nfp,
        stellsym=not args.non_stellsym,
    )


def _scalar_composed_geometry(args: argparse.Namespace, dofs: jax.Array):
    spec = _surface_spec_from_dofs(args, dofs)
    return (
        surface_rz_fourier_gamma_from_spec(spec),
        surface_rz_fourier_gammadash1_from_spec(spec),
        surface_rz_fourier_gammadash2_from_spec(spec),
    )


def _fused_geometry(args: argparse.Namespace, dofs: jax.Array):
    return surface_rz_fourier_geometry_from_spec(_surface_spec_from_dofs(args, dofs))


def _fused_geometry_vector(args: argparse.Namespace, dofs: jax.Array):
    return jnp.concatenate([jnp.ravel(part) for part in _fused_geometry(args, dofs)])


def _gamma_only(args: argparse.Namespace, dofs: jax.Array):
    return surface_rz_fourier_gamma_from_spec(_surface_spec_from_dofs(args, dofs))


def _hlo_stats(text: str) -> dict[str, int]:
    return {
        "cosine": len(re.findall(r"\bcosine(?:\(|\b)", text)),
        "sine": len(re.findall(r"\bsine(?:\(|\b)", text)),
        "reduce": len(re.findall(r"\breduce(?:\(|\b)", text)),
        "fusion": len(re.findall(r"\bfusion(?:\(|\b)", text)),
        "line_count": text.count("\n") + 1,
        "byte_count": len(text.encode("utf-8")),
    }


def _block_until_ready(tree) -> None:
    jax.tree.map(lambda value: value.block_until_ready(), tree)


def _compile_and_measure(
    label: str,
    fn: Callable[[jax.Array], object],
    dofs: jax.Array,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(dofs)
    lower_s = time.perf_counter() - start
    lowered_hlo_stats = _hlo_stats(lowered.as_text())

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    optimized_hlo_stats = _hlo_stats(compiled.as_text())
    for _ in range(warmup):
        _block_until_ready(compiled(dofs))
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        _block_until_ready(compiled(dofs))
        samples.append(time.perf_counter() - start)
    return {
        "label": label,
        "lower_s": lower_s,
        "compile_s": compile_s,
        "lowered_hlo": lowered_hlo_stats,
        "optimized_hlo": optimized_hlo_stats,
        "hlo": optimized_hlo_stats,
        "warm_steady_state_s": {
            "median": float(np.median(samples)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "samples": samples,
        },
    }


def _hlo_only(fn: Callable[[jax.Array], object], dofs: jax.Array) -> dict[str, object]:
    lowered = jax.jit(fn).lower(dofs)
    compiled = lowered.compile()
    optimized_hlo = _hlo_stats(compiled.as_text())
    return {
        "lowered_hlo": _hlo_stats(lowered.as_text()),
        "optimized_hlo": optimized_hlo,
        "hlo": optimized_hlo,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator)


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    dofs = _probe_dofs(args)
    composed = _compile_and_measure(
        "scalar_composed",
        lambda x: _scalar_composed_geometry(args, x),
        dofs,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    fused = _compile_and_measure(
        "fused",
        lambda x: _fused_geometry(args, x),
        dofs,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    composed_median = composed["warm_steady_state_s"]["median"]
    fused_median = fused["warm_steady_state_s"]["median"]
    line_ratio = _ratio(
        fused["optimized_hlo"]["line_count"],
        composed["optimized_hlo"]["line_count"],
    )
    time_ratio = _ratio(fused_median, composed_median)
    lowered_trig_reduce_keys = ("cosine", "sine", "reduce")
    fused_lowered_counts_lower = all(
        fused["lowered_hlo"][key] < composed["lowered_hlo"][key]
        for key in lowered_trig_reduce_keys
    )
    fused_optimized_counts_not_higher = all(
        fused["optimized_hlo"][key] <= composed["optimized_hlo"][key]
        for key in lowered_trig_reduce_keys
    )
    scalar_api_guard = _build_scalar_api_guard(args, dofs)
    hlo_gate_passed = (
        fused["optimized_hlo"]["line_count"] < composed["optimized_hlo"]["line_count"]
        and fused_lowered_counts_lower
        and fused_optimized_counts_not_higher
    )
    return {
        "shape": {
            "mpol": args.mpol,
            "ntor": args.ntor,
            "nfp": args.nfp,
            "nphi": args.nphi,
            "ntheta": args.ntheta,
            "stellsym": not args.non_stellsym,
            "dof_count": int(dofs.size),
        },
        "runtime": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "composed": composed,
        "fused": fused,
        "scalar_api_guard": scalar_api_guard,
        "comparison": {
            "fused_to_composed_hlo_line_ratio": line_ratio,
            "hlo_line_reduction_fraction": 1.0 - line_ratio,
            "fused_to_composed_time_ratio": time_ratio,
            "steady_state_speedup_fraction": 1.0 - time_ratio,
            "fused_hlo_line_count_lower": (
                fused["optimized_hlo"]["line_count"]
                < composed["optimized_hlo"]["line_count"]
            ),
            "fused_optimized_hlo_line_count_lower": (
                fused["optimized_hlo"]["line_count"]
                < composed["optimized_hlo"]["line_count"]
            ),
            "fused_lowered_trig_reduce_counts_lower": (fused_lowered_counts_lower),
            "fused_trig_reduce_counts_not_higher": (fused_optimized_counts_not_higher),
            "fused_optimized_trig_reduce_counts_not_higher": (
                fused_optimized_counts_not_higher
            ),
            "scalar_api_hlo_guard_passed": scalar_api_guard["passed"],
            "hlo_gate_passed": hlo_gate_passed,
        },
    }


def _build_scalar_api_guard(
    args: argparse.Namespace,
    dofs: jax.Array,
) -> dict[str, object]:
    gamma_stats = _hlo_only(lambda x: _gamma_only(args, x), dofs)
    geometry_stats = _hlo_only(lambda x: _fused_geometry_vector(args, x), dofs)
    gamma_jacfwd_stats = _hlo_only(
        lambda x: jax.jacfwd(lambda y: jnp.ravel(_gamma_only(args, y)))(x),
        dofs,
    )
    geometry_jacfwd_stats = _hlo_only(
        lambda x: jax.jacfwd(lambda y: _fused_geometry_vector(args, y))(x),
        dofs,
    )
    gamma_single_output_passed = (
        gamma_stats["optimized_hlo"]["line_count"]
        < geometry_stats["optimized_hlo"]["line_count"]
    )
    gamma_derivative_single_output_passed = (
        gamma_jacfwd_stats["optimized_hlo"]["line_count"]
        < geometry_jacfwd_stats["optimized_hlo"]["line_count"]
    )
    return {
        "gamma": gamma_stats,
        "geometry": geometry_stats,
        "gamma_jacfwd": gamma_jacfwd_stats,
        "geometry_jacfwd": geometry_jacfwd_stats,
        "gamma_single_output_passed": gamma_single_output_passed,
        "gamma_derivative_single_output_passed": (
            gamma_derivative_single_output_passed
        ),
        "passed": (
            gamma_single_output_passed and gamma_derivative_single_output_passed
        ),
    }


def main() -> None:
    args = parse_args()
    payload = run_probe(args)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    comparison = payload["comparison"]
    if args.fail_on_regression and not (
        comparison["hlo_gate_passed"] and comparison["scalar_api_hlo_guard_passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
