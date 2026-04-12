from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

import simsopt.config as simsopt_config  # type: ignore[import-untyped]
from simsopt.geo.optimizer_jax import (  # type: ignore[import-untyped]
    _mark_cacheable_jit_value_and_grad,
    jax_minimize,
    private_optimizer_runtime_is_supported,
    target_minimize,
)

OptimizerMethod = Literal["lbfgs-ondevice", "bfgs-ondevice"]


def _configure_strict_cpu_parity_backend() -> bool:
    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    return private_optimizer_runtime_is_supported(jax.__version__)


class _CompileCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "Compiling jit(run_solver)" in record.getMessage():
            self.count += 1


def _assert_run_solver_compiles_once(run_once) -> None:
    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            for _ in range(3):
                run_once()
        assert handler.count == 1, handler.count
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _run_compile_count_case(method: OptimizerMethod) -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad(x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(vector, vector)

    cacheable_quad = _mark_cacheable_jit_value_and_grad(quad)
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def run_once() -> None:
        result = jax_minimize(cacheable_quad, x0, method=method, maxiter=5)
        assert result.success is True

    _assert_run_solver_compiles_once(run_once)


def _run_target_compile_count_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad_value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        vector = jnp.asarray(x, dtype=jnp.float64)
        value = half * jnp.dot(vector, vector)
        grad = vector
        return value, grad

    cacheable_quad_value_and_grad = _mark_cacheable_jit_value_and_grad(
        jax.jit(quad_value_and_grad)
    )
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def run_once() -> None:
        result = target_minimize(
            cacheable_quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            maxiter=5,
        )
        assert result.success is True

    _assert_run_solver_compiles_once(run_once)


class _ShiftedQuadratic:
    def __init__(self, target: Sequence[float]) -> None:
        self.target = np.asarray(tuple(target), dtype=np.float64)
        self.half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def __call__(self, x: jax.Array) -> jax.Array:
        vector = jnp.asarray(x, dtype=jnp.float64)
        target = jnp.asarray(self.target, dtype=jnp.float64)
        diff = vector - target
        return self.half * jnp.dot(diff, diff)


def _run_mutable_objective_state_case() -> None:
    if not _configure_strict_cpu_parity_backend():
        return

    objective = _ShiftedQuadratic([0.0, 0.0])
    x0 = jnp.asarray(np.array([2.0, -1.0], dtype=np.float64))

    first = jax_minimize(objective, x0, method="bfgs-ondevice", maxiter=20)
    objective.target = np.asarray([1.5, -0.5], dtype=np.float64)
    second = jax_minimize(objective, x0, method="bfgs-ondevice", maxiter=20)

    np.testing.assert_allclose(
        np.asarray(first.x),
        np.asarray([0.0, 0.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(second.x),
        np.asarray([1.5, -0.5]),
        atol=1e-6,
    )


def _parse_optimizer_method(method: str) -> OptimizerMethod:
    if method == "lbfgs-ondevice":
        return cast(OptimizerMethod, method)
    if method == "bfgs-ondevice":
        return cast(OptimizerMethod, method)
    raise ValueError(f"unsupported optimizer method {method!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process-isolated JAX runtime regression cases.",
    )
    subparsers = parser.add_subparsers(dest="case", required=True)

    compile_count = subparsers.add_parser("compile-count")
    compile_count.add_argument(
        "method",
        choices=("lbfgs-ondevice", "bfgs-ondevice"),
    )
    subparsers.add_parser("target-compile-count")
    subparsers.add_parser("mutable-objective-state")

    args = parser.parse_args(argv)

    if args.case == "compile-count":
        _run_compile_count_case(_parse_optimizer_method(args.method))
        return 0
    if args.case == "target-compile-count":
        _run_target_compile_count_case()
        return 0
    if args.case == "mutable-objective-state":
        _run_mutable_objective_state_case()
        return 0
    raise ValueError(f"unsupported subprocess case {args.case!r}")


if __name__ == "__main__":
    raise SystemExit(main())
