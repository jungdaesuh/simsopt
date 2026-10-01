"""Behaviour of the JAX host/device boundary owners.

``simsopt_jax.runtime.host_boundary`` owns the strict transfer guard, its one
permit (``allow_host_transfers``) and the host readers;
``simsopt_jax.backend.dtypes`` owns placement. Code under
``disallow_host_transfers()`` moves data only through these owners, so each
must work under the strict guard, and the permit must lift the guard for its
own block and no further.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.backend.dtypes import explicit_device_array, runtime_device_put_tree
from simsopt_jax.runtime.host_boundary import (
    allow_host_transfers,
    block_until_ready,
    disallow_host_transfers,
    host_array,
    host_tree_after_ready,
)


def test_allow_host_transfers_lifts_the_strict_guard_for_its_block_only() -> None:
    device_array = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    host_values = np.asarray([3.0, 4.0], dtype=np.float64)

    with disallow_host_transfers():
        with pytest.raises(jax.errors.JaxRuntimeError):
            device_array + host_values
        with allow_host_transfers():
            permitted = device_array + host_values
        with pytest.raises(jax.errors.JaxRuntimeError):
            device_array + host_values

    np.testing.assert_array_equal(jax.device_get(permitted), [4.0, 6.0])


def test_placement_owners_place_host_values_under_the_strict_guard() -> None:
    tree = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "integer": (np.asarray(3, dtype=np.int16),),
    }
    reference = jax.device_put(np.zeros(2, dtype=np.float64))

    with disallow_host_transfers():
        placed_tree = runtime_device_put_tree(tree)
        placed_array = explicit_device_array(
            [5.0, 6.0], dtype=jnp.float64, reference=reference
        )

    assert isinstance(placed_tree["float"], jax.Array)
    np.testing.assert_array_equal(jax.device_get(placed_tree["float"]), tree["float"])
    assert jax.device_get(placed_tree["integer"][0]).item() == 3
    assert placed_array.dtype == jnp.float64
    assert placed_array.sharding == reference.sharding
    np.testing.assert_array_equal(jax.device_get(placed_array), [5.0, 6.0])


def test_host_readers_return_writeable_host_copies_under_the_strict_guard() -> None:
    value = {
        "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "scalar": (jnp.asarray(3, dtype=jnp.int32),),
    }

    with disallow_host_transfers():
        array = host_array(value["vector"], dtype=np.float32)
        tree = host_tree_after_ready(value)

    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert array.flags.writeable
    np.testing.assert_array_equal(array, [1.0, 2.0])
    assert isinstance(tree["vector"], np.ndarray)
    assert tree["vector"].flags.writeable
    assert tree["scalar"][0].item() == 3


def test_runtime_device_put_tree_preserves_structure_and_exact_leaf_dtypes() -> None:
    value = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "integer": (np.asarray(3, dtype=np.int16),),
    }

    placed = runtime_device_put_tree(value)

    assert placed.keys() == value.keys()
    assert placed["float"].dtype == jnp.float32
    assert placed["integer"][0].dtype == jnp.int16


def test_readiness_and_host_tree_preserve_pytree_structure() -> None:
    value = {
        "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "scalar": (jnp.asarray(3, dtype=jnp.int32),),
    }

    ready = block_until_ready(value)
    host = host_tree_after_ready(ready)

    assert host.keys() == value.keys()
    np.testing.assert_array_equal(host["vector"], np.asarray([1.0, 2.0]))
    assert host["scalar"][0].item() == 3
    assert host["vector"].flags.writeable
