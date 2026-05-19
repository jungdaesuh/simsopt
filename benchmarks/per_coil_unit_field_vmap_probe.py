"""Scaling probe for per-coil unit-field vectorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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

from simsopt.field.biotsavart_jax_backend import _per_coil_unit_field
from simsopt.jax_core.biotsavart import biot_savart_B
from simsopt.jax_core.specs import CoilGroupSpec, GroupedCoilSetSpec


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncoils", default="2,4,8,16")
    parser.add_argument("--npoints", type=int, default=256)
    parser.add_argument("--nquad", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def _parse_ncoils(raw: str) -> tuple[int, ...]:
    values = tuple(int(field) for field in raw.split(",") if field.strip())
    if len(values) < 2:
        raise ValueError("--ncoils must list at least two positive sizes.")
    if any(value <= 0 for value in values):
        raise ValueError("--ncoils entries must be positive.")
    return values


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
    result = run_probe(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
