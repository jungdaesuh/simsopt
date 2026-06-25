from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from simsopt_jax.core.sharding import (
    SeedBatchShardingConfig,
    SurfaceQuadratureShardingConfig,
    TrajectoryBatchShardingConfig,
    maybe_shard_seed_batch_inputs,
    maybe_shard_surface_quadrature_inputs,
    maybe_shard_trajectory_batch_inputs,
    seed_batch_sharding_summary,
    surface_quadrature_sharding_summary,
    trajectory_batch_sharding_summary,
)


def _single_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1], dtype=object), ("batch",))


@pytest.mark.parametrize(
    ("config_cls", "maybe_shard_fn", "summary_fn", "active_key", "device_count_key"),
    [
        (
            TrajectoryBatchShardingConfig,
            maybe_shard_trajectory_batch_inputs,
            trajectory_batch_sharding_summary,
            "trajectory_sharded",
            "trajectory_device_count",
        ),
        (
            SeedBatchShardingConfig,
            maybe_shard_seed_batch_inputs,
            seed_batch_sharding_summary,
            "seed_batch_sharded",
            "seed_batch_device_count",
        ),
        (
            SurfaceQuadratureShardingConfig,
            maybe_shard_surface_quadrature_inputs,
            surface_quadrature_sharding_summary,
            "surface_quadrature_sharded",
            "surface_quadrature_device_count",
        ),
    ],
)
def test_leading_axis_sharding_configs_preserve_public_contract(
    config_cls,
    maybe_shard_fn,
    summary_fn,
    active_key,
    device_count_key,
):
    assert is_dataclass(config_cls)

    config = config_cls(
        mesh=_single_device_mesh(),
        axis_name="batch",
        device_count=1,
        strategy="hybrid",
    )
    assert type(config) is config_cls
    assert [field.name for field in fields(config)] == [
        "mesh",
        "axis_name",
        "device_count",
        "strategy",
    ]

    values = jnp.zeros((2, 3), dtype=jnp.float64)
    unsharded_summary = summary_fn(values, config=None)
    assert unsharded_summary[active_key] is False
    assert device_count_key not in unsharded_summary

    (placed_values,) = maybe_shard_fn(values, config=config)
    sharded_summary = summary_fn(placed_values, config=config)
    assert sharded_summary[active_key] is True
    assert sharded_summary["axis"] == "batch"
    assert sharded_summary["strategy"] == "hybrid"
    assert sharded_summary["mesh_shape"] == {"batch": 1}
    assert sharded_summary[device_count_key] == 1
    assert "batch" in str(sharded_summary["spec"])
