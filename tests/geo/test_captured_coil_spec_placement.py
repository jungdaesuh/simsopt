"""A captured coil spec must be host-resident; an operand spec must not be.

``simsopt_jax.core.specs.host_resident_spec`` states the rule once. XLA turns a
captured concrete ``jax.Array`` into an MLIR literal by copying it back to the
host, once per lowering, and ``jax.transfer_guard("disallow")`` refuses that
copy on a real device. The traceable surface-objective adapter hostifies the
trees it captures; it must apply that one rule, not a second version of it.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.core.specs import host_resident_spec
from simsopt_jax_adapters.geo.surface_objectives import (
    _traceable_runtime_hostify_tree,
)


def test_traceable_runtime_hostify_tree_is_the_host_resident_spec_rule():
    """The adapter's hostify helper states no second version of the rule."""
    tree = {
        "device": jnp.asarray(np.arange(6, dtype=np.float64).reshape(2, 3)),
        "host": np.linspace(0.0, 1.0, 4, dtype=np.float64),
        "scalar": 2.5,
        "flag": True,
    }

    hostified = _traceable_runtime_hostify_tree(tree)
    canonical = host_resident_spec(tree)

    assert not [
        leaf for leaf in jax.tree.leaves(hostified) if isinstance(leaf, jax.Array)
    ]
    assert jax.tree.structure(hostified) == jax.tree.structure(canonical)
    for key in ("device", "host"):
        np.testing.assert_array_equal(
            np.asarray(hostified[key]),
            np.asarray(canonical[key]),
            err_msg=f"hostify diverged from host_resident_spec on {key!r}",
        )
    assert hostified["scalar"] == canonical["scalar"] == 2.5
    assert hostified["flag"] is canonical["flag"] is True
