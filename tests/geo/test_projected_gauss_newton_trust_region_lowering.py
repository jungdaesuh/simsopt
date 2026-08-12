"""Compile-size regression gate for the step-bound safeguard backtrack loop.

``run_projected_gauss_newton_trust_region_loop`` previously trace-time-unrolled
the ``options.enable_step_bound_safeguard`` backtrack loop
(``for _ in range(_STEP_BOUND_SAFEGUARD_BACKTRACKS)``), tripling the lowered
IR relative to the safeguard-off baseline and causing a ~70-minute XLA:CPU
compile in production. The fix rolls that loop into ``lax.fori_loop`` so the
lowered IR size stops scaling with the unrolled backtrack count and stops
scaling at all with ``maximum_attempts``.

Baseline measured 2026-08-12 at commit e8977b2a0 on the UNROLLED code, using
the tiny 2-dof fixture below:

    safeguard off / attempts=4:    324,568 IR bytes
    safeguard off / attempts=300:  328,156 IR bytes
    safeguard on  / attempts=4:    971,837 IR bytes  (ratio on/off = 2.99)
    safeguard on  / attempts=300:  980,289 IR bytes

These tests pin the rolled-loop invariants: the safeguard must not blow up
the IR relative to safeguard-off, and the IR size must not grow with the
attempt budget.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonOptions,
    run_projected_gauss_newton_trust_region_loop,
)

jax.config.update("jax_enable_x64", True)

_X0 = jnp.asarray([1.0, 0.0], dtype=jnp.float64)


def _linear_equality_joint(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    return 0.5 * values[0] ** 2, jnp.reshape(values[1], (1,))


def _linear_objective_residual(values: jax.Array) -> jax.Array:
    return jnp.reshape(values[0], (1,))


def _lowered_ir_size(*, enable_step_bound_safeguard: bool, attempts: int) -> int:
    options = ProjectedGaussNewtonOptions(
        maximum_accepted_steps=min(3, attempts),
        maximum_attempts=attempts,
        initial_trust_radius=0.25,
        maximum_trust_radius=1.0,
        enable_step_bound_safeguard=enable_step_bound_safeguard,
    )

    def loop(coordinates: jax.Array):
        return run_projected_gauss_newton_trust_region_loop(
            _linear_equality_joint,
            _linear_objective_residual,
            coordinates,
            options=options,
        )

    return len(jax.jit(loop).lower(_X0).as_text())


def test_step_bound_safeguard_lowering_stays_rolled() -> None:
    off_size = _lowered_ir_size(enable_step_bound_safeguard=False, attempts=4)
    on_size = _lowered_ir_size(enable_step_bound_safeguard=True, attempts=4)
    ratio = on_size / off_size

    assert ratio < 1.7, (
        "step-bound safeguard backtrack loop regressed to trace-time unrolling: "
        f"on/off lowered-IR ratio at attempts=4 is {ratio:.3f} (on={on_size}, "
        f"off={off_size}); unrolled baseline measured 2.99, rolled fori_loop "
        "must stay well below 1.7"
    )


def test_lowering_size_is_attempt_budget_independent() -> None:
    small_size = _lowered_ir_size(enable_step_bound_safeguard=True, attempts=4)
    large_size = _lowered_ir_size(enable_step_bound_safeguard=True, attempts=300)

    assert large_size < 1.05 * small_size, (
        "safeguard-on lowered IR grows with maximum_attempts: "
        f"attempts=300 is {large_size} bytes vs attempts=4 at {small_size} bytes; "
        "the outer attempt loop must stay a lax.fori_loop, not become "
        "unrolled or scanned with unroll>1"
    )


def test_safeguard_off_lowering_baseline_is_stable() -> None:
    off_size = _lowered_ir_size(enable_step_bound_safeguard=False, attempts=4)

    # 2026-08-12 baseline (safeguard off, attempts=4): ~324,568 IR bytes.
    assert off_size < 600_000, (
        "safeguard-off lowered IR ballooned past the loose absolute tripwire: "
        f"{off_size} bytes >= 600,000; the 2026-08-12 baseline is ~325k, so "
        "this points at the whole loop body silently doubling"
    )
