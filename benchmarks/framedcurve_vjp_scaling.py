"""Scaling probe for framed-curve composed VJP helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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

from simsopt.geo import framedcurve_jax as fcj


jax.config.update("jax_enable_x64", True)


CaseFn = Callable[..., tuple[jax.Array, ...]]
ScalarFn = Callable[..., jax.Array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nquad", default="32,64,128")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def _parse_nquad(raw: str) -> tuple[int, ...]:
    values = tuple(int(field) for field in raw.split(",") if field.strip())
    if not values:
        raise ValueError("--nquad must list at least one positive size.")
    if any(value <= 0 for value in values):
        raise ValueError("--nquad entries must be positive.")
    return values


def _curve_inputs(nquad: int) -> tuple[jax.Array, ...]:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, nquad, endpoint=False)
    gamma = jnp.stack(
        (
            jnp.cos(theta),
            jnp.sin(theta),
            0.12 * jnp.sin(2.0 * theta),
        ),
        axis=1,
    )
    gammadash = jnp.stack(
        (
            -jnp.sin(theta),
            jnp.cos(theta),
            0.24 * jnp.cos(2.0 * theta),
        ),
        axis=1,
    )
    gammadashdash = jnp.stack(
        (
            -jnp.cos(theta),
            -jnp.sin(theta),
            -0.48 * jnp.sin(2.0 * theta),
        ),
        axis=1,
    )
    gammadashdashdash = jnp.stack(
        (
            jnp.sin(theta),
            -jnp.cos(theta),
            -0.96 * jnp.cos(2.0 * theta),
        ),
        axis=1,
    )
    alpha = 0.17 * jnp.sin(theta)
    alphadash = 0.17 * jnp.cos(theta)
    cotangent = 0.5 + jnp.cos(theta)
    return (
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
        cotangent,
    )


def _legacy_isolated_vjps(
    scalar_fn: ScalarFn,
    primals: tuple[jax.Array, ...],
    cotangent: jax.Array,
) -> tuple[jax.Array, ...]:
    gradients = []
    for primal_index, primal in enumerate(primals):
        _, pullback = jax.vjp(
            lambda value: scalar_fn(
                *primals[:primal_index],
                value,
                *primals[primal_index + 1 :],
            ),
            primal,
        )
        gradients.append(pullback(cotangent)[0])
    return tuple(gradients)


def _frame_twist_current(gammadash, t, n, ndash, cotangent):
    return fcj._frame_twist_vjps(gammadash, t, n, ndash, cotangent)


def _frame_twist_legacy(gammadash, t, n, ndash, cotangent):
    return _legacy_isolated_vjps(
        fcj._frame_twist,
        (gammadash, t, n, ndash),
        cotangent,
    )


def _centroid_torsion_current(
    gamma,
    gammadash,
    gammadashdash,
    alpha,
    alphadash,
    cotangent,
):
    return fcj._centroid_torsion_vjps(
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
        cotangent,
    )


def _centroid_torsion_legacy(
    gamma,
    gammadash,
    gammadashdash,
    alpha,
    alphadash,
    cotangent,
):
    return _legacy_isolated_vjps(
        fcj._torsion_centroid,
        (gamma, gammadash, gammadashdash, alpha, alphadash),
        cotangent,
    )


def _centroid_binormal_current(
    gamma,
    gammadash,
    gammadashdash,
    alpha,
    alphadash,
    cotangent,
):
    return fcj._centroid_binormal_curvature_vjps(
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
        cotangent,
    )


def _centroid_binormal_legacy(
    gamma,
    gammadash,
    gammadashdash,
    alpha,
    alphadash,
    cotangent,
):
    return _legacy_isolated_vjps(
        fcj._binormal_curvature_centroid,
        (gamma, gammadash, gammadashdash, alpha, alphadash),
        cotangent,
    )


def _frenet_torsion_current(
    gamma,
    gammadash,
    gammadashdash,
    gammadashdashdash,
    alpha,
    alphadash,
    cotangent,
):
    return fcj._frenet_torsion_vjps(
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
        cotangent,
    )


def _frenet_torsion_legacy(
    gamma,
    gammadash,
    gammadashdash,
    gammadashdashdash,
    alpha,
    alphadash,
    cotangent,
):
    return _legacy_isolated_vjps(
        fcj._torsion_frenet,
        (gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash),
        cotangent,
    )


def _frenet_binormal_current(
    gamma,
    gammadash,
    gammadashdash,
    gammadashdashdash,
    alpha,
    alphadash,
    cotangent,
):
    return fcj._frenet_binormal_curvature_vjps(
        gamma,
        gammadash,
        gammadashdash,
        gammadashdashdash,
        alpha,
        alphadash,
        cotangent,
    )


def _frenet_binormal_legacy(
    gamma,
    gammadash,
    gammadashdash,
    gammadashdashdash,
    alpha,
    alphadash,
    cotangent,
):
    return _legacy_isolated_vjps(
        fcj._binormal_curvature_frenet,
        (gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash),
        cotangent,
    )


CaseArgs = Callable[[tuple[jax.Array, ...]], tuple[jax.Array, ...]]


def _frame_twist_args(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return inputs[1:4] + (inputs[2] + 0.1 * inputs[1], inputs[6])


def _centroid_args(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], inputs[6]


CASES: tuple[tuple[str, CaseFn, CaseFn, CaseArgs], ...] = (
    (
        "frame_twist",
        _frame_twist_current,
        _frame_twist_legacy,
        _frame_twist_args,
    ),
    (
        "centroid_torsion",
        _centroid_torsion_current,
        _centroid_torsion_legacy,
        _centroid_args,
    ),
    (
        "centroid_binormal_curvature",
        _centroid_binormal_current,
        _centroid_binormal_legacy,
        _centroid_args,
    ),
    (
        "frenet_torsion",
        _frenet_torsion_current,
        _frenet_torsion_legacy,
        lambda inputs: inputs,
    ),
    (
        "frenet_binormal_curvature",
        _frenet_binormal_current,
        _frenet_binormal_legacy,
        lambda inputs: inputs,
    ),
)


def _block_ready(value):
    return jax.block_until_ready(value)


def _time_call(
    fn: CaseFn,
    args: tuple[jax.Array, ...],
    *,
    warmup: int,
    repeat: int,
) -> float:
    compiled = jax.jit(fn)
    for _ in range(warmup):
        _block_ready(compiled(*args))
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        _block_ready(compiled(*args))
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _eqn_count(fn: CaseFn, args: tuple[jax.Array, ...]) -> int:
    return len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)


def _assert_outputs_match(
    current_fn: CaseFn,
    legacy_fn: CaseFn,
    args: tuple[jax.Array, ...],
) -> None:
    current = current_fn(*args)
    legacy = legacy_fn(*args)
    for current_leaf, legacy_leaf in zip(
        jax.tree.leaves(current),
        jax.tree.leaves(legacy),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(current_leaf),
            np.asarray(legacy_leaf),
            rtol=1e-12,
            atol=1e-12,
        )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for nquad in _parse_nquad(args.nquad):
        base_inputs = _curve_inputs(nquad)
        for name, current_fn, legacy_fn, select_args in CASES:
            case_args = select_args(base_inputs)
            _assert_outputs_match(current_fn, legacy_fn, case_args)
            current_s = _time_call(
                current_fn,
                case_args,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            legacy_s = _time_call(
                legacy_fn,
                case_args,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            rows.append(
                {
                    "case": name,
                    "nquad": nquad,
                    "current_median_s": current_s,
                    "legacy_median_s": legacy_s,
                    "speedup": legacy_s / current_s,
                    "current_jaxpr_eqns": _eqn_count(current_fn, case_args),
                    "legacy_jaxpr_eqns": _eqn_count(legacy_fn, case_args),
                }
            )
    if args.fail_on_regression:
        regressions = [
            row
            for row in rows
            if row["current_jaxpr_eqns"] >= row["legacy_jaxpr_eqns"]
        ]
        if regressions:
            raise RuntimeError(json.dumps({"regressions": regressions}, indent=2))
    return {"rows": rows}


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
