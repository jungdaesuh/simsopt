"""A captured coil spec must be host-resident; an operand spec must not be.

``simsopt_jax.core.specs.host_resident_spec`` states the rule once. XLA turns a
captured concrete ``jax.Array`` into an MLIR literal by copying it back to the
host, once per lowering, and ``jax.transfer_guard("disallow")`` refuses that
copy on a real device. ``nested_ls_runtime_coil_closures`` takes the coil DOFs
as a program *argument* but rebuilds the grouped spec inside the traced body, so
the frozen reconstruction template is a *capture*: before the conversion the
two jitted kernels each lowered 39 device-array constants -- 18 current
templates, 15 symmetry rotation matrices, 3 curve-DOF templates and 3 curve
quadpoint grids, i.e. the whole coil graph.

The constant count is the observable this pins, not the placement of any one
leaf, because it is what lowering actually does and it is visible on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import array as jax_array_mod
from jax._src.interpreters import mlir
from simsopt_jax.core.specs import host_resident_spec
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_runtime_coil_closures,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    _traceable_runtime_hostify_tree,
)

from .test_nested_ls_reduced import _nested_ls_volume_pair


def _lowered_device_array_constants(monkeypatch, lower):
    """Return the shape/dtype of every device array lowered as an MLIR literal."""
    recorded: dict[int, tuple[tuple[int, ...], str]] = {}
    original = mlir._constant_handlers[jax_array_mod.ArrayImpl]

    def handler(value, *args, **kwargs):
        recorded.setdefault(id(value), (tuple(value.shape), str(value.dtype)))
        return original(value, *args, **kwargs)

    monkeypatch.setitem(
        mlir._constant_handlers, jax_array_mod.ArrayImpl, handler
    )
    lower()
    return sorted(recorded.values())


@pytest.mark.boozer
def test_runtime_coil_closures_capture_no_device_array_constants(monkeypatch):
    """The captured reconstruction template may not be read back at lowering."""
    _native, jax_boozer, decision, _iota, _g = _nested_ls_volume_pair()
    residual_fn, objective_fn, _phi_hat = nested_ls_runtime_coil_closures(jax_boozer)
    packed = jnp.asarray(np.asarray(decision, dtype=np.float64))
    coil_dofs = jnp.asarray(
        np.asarray(jax.device_get(jax_boozer.biotsavart.x), dtype=np.float64)
    )

    for name, closure in (("objective", objective_fn), ("residual", residual_fn)):
        constants = _lowered_device_array_constants(
            monkeypatch,
            lambda closure=closure: jax.jit(closure).lower(packed, coil_dofs),
        )
        assert constants == [], (
            f"the runtime-coil {name} kernel lowered {len(constants)} device-array "
            f"constants {constants}; every one is a device-to-host copy per "
            "lowering that jax.transfer_guard('disallow') refuses on GPU"
        )


@pytest.mark.boozer
def test_runtime_coil_closures_keep_the_biotsavart_spec_device_resident(monkeypatch):
    """The maker's own spec stays device-resident for its argument-role callers."""
    del monkeypatch
    _native, jax_boozer, _decision, _iota, _g = _nested_ls_volume_pair()
    nested_ls_runtime_coil_closures(jax_boozer)

    leaves = jax.tree.leaves(jax_boozer.biotsavart.coil_dof_extraction_spec())
    assert leaves
    off_device = [leaf for leaf in leaves if not isinstance(leaf, jax.Array)]
    assert not off_device, (
        f"building the runtime-coil closures left {len(off_device)} leaf/leaves of "
        "BiotSavartJAX's own extraction spec on the host; the stage-two examples "
        "pass that spec to jax.jit as an operand, where a host leaf is an "
        "implicit host-to-device transfer"
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
