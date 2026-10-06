"""Focused subprocess cases for core JAX finite-difference regressions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from import_smoke_cases import prefer_local_simsopt_source_tree


prefer_local_simsopt_source_tree()

from simsopt_jax.core._finite_difference import (
    forward_jacobian_shard_map,
    forward_jacobian_shard_map_columns,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _run_forward_jacobian_shard_map_two_device_case() -> None:
    devices = jax.devices()
    assert len(devices) >= 2, devices
    mesh = Mesh(np.asarray(devices[:2]), ("dof",))
    replicated = NamedSharding(mesh, P())
    column_sharding = NamedSharding(mesh, P("dof"))
    matrix = jax.device_put(
        jnp.asarray(
            ((1.0, 2.0, 3.0), (-1.0, 0.5, 4.0)),
            dtype=jnp.float64,
        ),
        replicated,
    )

    def residual(x: jax.Array) -> jax.Array:
        return matrix @ x

    compiled = jax.jit(
        lambda x: forward_jacobian_shard_map(
            residual,
            x,
            abs_step=2.0**-30,
            mesh=mesh,
        )
    )
    x0 = jax.device_put(
        jnp.asarray([0.2, -0.3, 0.4], dtype=jnp.float64),
        replicated,
    )
    selected_columns = jax.device_put(
        jnp.asarray([0, 2], dtype=jnp.int32),
        column_sharding,
    )
    compiled_selected = jax.jit(
        lambda x, columns: forward_jacobian_shard_map_columns(
            residual,
            x,
            columns,
            abs_step=2.0**-30,
            mesh=mesh,
        )
    )
    compiled(x0).block_until_ready()
    compiled_selected(x0, selected_columns).block_until_ready()

    with jax.transfer_guard("disallow"):
        jacobian = compiled(x0)
        selected_jacobian = compiled_selected(x0, selected_columns)
        jacobian.block_until_ready()
        selected_jacobian.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(matrix),
        rtol=5.0e-7,
        atol=5.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(selected_jacobian),
        np.asarray(matrix)[:, (0, 2)],
        rtol=5.0e-7,
        atol=5.0e-7,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=("forward-jacobian-shard-map-two-device",),
    )
    args = parser.parse_args(argv)
    if args.case == "forward-jacobian-shard-map-two-device":
        _run_forward_jacobian_shard_map_two_device_case()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
