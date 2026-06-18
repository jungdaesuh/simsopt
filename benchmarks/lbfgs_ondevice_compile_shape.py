"""Compile-shape diagnostic for the stepwise L-BFGS-B control kernels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

jax.config.update("jax_enable_x64", True)

from benchmarks.traceable_compile_shape import (
    lower_to_text,
    summarize_lowered_text,
)
from simsopt_jax.geo.optimizers.private import _lbfgs as private_lbfgs
from simsopt_jax.geo.optimizers.private import _lbfgsb_scipy as lbfgsb


def _quadratic_value_and_grad(x):
    vector = jnp.asarray(x, dtype=jnp.float64)
    return 0.5 * jnp.dot(vector, vector), vector


def _jaxpr_summary(fn, *args) -> dict[str, int]:
    closed_jaxpr = jax.make_jaxpr(fn)(*args)
    jaxpr_text = str(closed_jaxpr)
    return {
        "jaxpr_eqn_count": len(closed_jaxpr.jaxpr.eqns),
        "jaxpr_text_bytes": len(jaxpr_text.encode("utf-8")),
        "jaxpr_text_lines": len(jaxpr_text.splitlines()),
    }


def _lower_summary(label: str, fn, *args) -> dict[str, int | float | str | None]:
    measurement = lower_to_text(jax.jit(fn), *args)
    summary = summarize_lowered_text(
        label,
        measurement.lowered_text,
        lower_s=measurement.lower_s,
    )
    summary.update(_jaxpr_summary(fn, *args))
    return summary


def _result_payload(state, *, maxiter: int, maxfun: int):
    history = lbfgsb.lbfgsb_inverse_hessian_history(state)
    return private_lbfgs._lbfgsb_state_to_lbfgs_results(
        state,
        history=history,
        maxiter_limit=jnp.asarray(maxiter, dtype=jnp.int32),
        maxfun_limit=jnp.asarray(maxfun, dtype=jnp.int32),
    )


def _build_summaries(
    *,
    dimension: int,
    maxcor: int,
    maxiter: int,
    maxfun: int,
    maxls: int,
    ftol: float,
    gtol: float,
) -> list[dict[str, int | float | str | None]]:
    x0 = jnp.arange(1, dimension + 1, dtype=jnp.float64)

    def init_state(x):
        return lbfgsb.lbfgsb_initial_state(
            x,
            m=maxcor,
            bounds=None,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
        )

    state0 = init_state(x0)

    def step_kernel(state):
        return lbfgsb.lbfgsb_advance_to_next_observable(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    def result_kernel(state):
        return _result_payload(state, maxiter=maxiter, maxfun=maxfun)

    def monolithic_kernel(state):
        final_state = lbfgsb.lbfgsb_mainlb(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )
        return _result_payload(final_state, maxiter=maxiter, maxfun=maxfun)

    return [
        _lower_summary("init_state", init_state, x0),
        _lower_summary("step_to_next_observable", step_kernel, state0),
        _lower_summary("result_payload", result_kernel, state0),
        _lower_summary("old_monolithic_full_solve", monolithic_kernel, state0),
    ]


def _comparison(summaries: list[dict[str, int | float | str | None]]) -> dict[str, int]:
    by_label = {str(row["label"]): row for row in summaries}
    monolithic = by_label["old_monolithic_full_solve"]
    step = by_label["step_to_next_observable"]
    result = by_label["result_payload"]
    return {
        "old_monolithic_text_bytes": int(monolithic["text_bytes"]),
        "step_text_bytes": int(step["text_bytes"]),
        "result_payload_text_bytes": int(result["text_bytes"]),
        "step_plus_result_text_bytes": int(step["text_bytes"])
        + int(result["text_bytes"]),
        "old_monolithic_jaxpr_text_bytes": int(monolithic["jaxpr_text_bytes"]),
        "step_jaxpr_text_bytes": int(step["jaxpr_text_bytes"]),
        "result_payload_jaxpr_text_bytes": int(result["jaxpr_text_bytes"]),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write L-BFGS-B on-device compile-shape diagnostics."
    )
    parser.add_argument(
        "--output-json",
        default=".artifacts/lbfgs_ondevice_compile_shape_20260618.json",
        help="Path for the diagnostic JSON payload.",
    )
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--maxcor", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--maxfun", type=int, default=20)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=0.0)
    parser.add_argument("--gtol", type=float, default=1e-8)
    args = parser.parse_args()

    summaries = _build_summaries(
        dimension=args.dimension,
        maxcor=args.maxcor,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        maxls=args.maxls,
        ftol=args.ftol,
        gtol=args.gtol,
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "case": {
            "objective": "deterministic_quadratic",
            "dimension": int(args.dimension),
            "maxcor": int(args.maxcor),
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxls": int(args.maxls),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "dtype": np.dtype(np.float64).str,
        },
        "summaries": summaries,
        "comparison": _comparison(summaries),
    }
    output_path = Path(args.output_json)
    _write_json(output_path, payload)
    print(json.dumps({"output_json": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
