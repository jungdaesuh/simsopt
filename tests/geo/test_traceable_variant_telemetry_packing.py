from __future__ import annotations

import jax.numpy as jnp
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _pack_traceable_forward_result,
)


def _packed(**extra):
    scalar = jnp.asarray(1.0, dtype=jnp.float64)
    return _pack_traceable_forward_result(
        value=scalar,
        x=jnp.ones(3, dtype=jnp.float64),
        sdofs=jnp.ones(1, dtype=jnp.float64),
        iota=scalar,
        G=scalar,
        linear_solve_factors=None,
        success=jnp.asarray(True),
        primal_success=jnp.asarray(True),
        adjoint_linear_solve_available=jnp.asarray(True),
        newton_trace_capacity=0,
        **extra,
    )


def test_c0_packed_forward_result_has_no_variant_keys() -> None:
    packed = _packed()
    explicit_empty = _packed(exact_newton_variant_telemetry={})

    assert not any(key.startswith("exact_newton_variant_") for key in packed)
    assert frozenset(packed) == frozenset(explicit_empty)


def test_variant_telemetry_is_carried_without_renaming_or_placeholders() -> None:
    telemetry = {
        "exact_newton_variant_dense_materialization_count": jnp.asarray(
            2, dtype=jnp.int32
        ),
        "exact_newton_variant_stop_reason_code": jnp.asarray(3, dtype=jnp.int32),
        "exact_newton_variant_numerical_failure": jnp.asarray(True),
        "exact_newton_variant_stalled": jnp.asarray(True),
    }

    packed = _packed(exact_newton_variant_telemetry=telemetry)

    for key, value in telemetry.items():
        assert packed[key] is value
