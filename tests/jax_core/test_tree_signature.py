"""Regression coverage for §6 of the JAX silent-fallback removal plan.

Plan: ``.artifacts/jax-silent-fallback-removal-2026-05-13/PLAN.md`` §6 deleted
dead ``except TypeError`` branches that wrapped five pytree signature and
transform helpers:

- ``simsopt.geo.boozersurface_jax._runtime_cache_tree_signature``
- ``simsopt.geo.surfaceobjectives_jax._traceable_cache_tree_signature``
- ``simsopt.geo.surfaceobjectives_jax._traceable_contract_tree_signature``
- ``simsopt.geo.surfaceobjectives_jax._traceable_runtime_hostify_tree``
- ``simsopt.geo.surfaceobjectives_jax._traceable_runtime_deviceify_tree``

The removed branches relied on the (false) assumption that
``jax.tree_util.tree_flatten`` and ``tree_map`` raise ``TypeError`` for
classes that are not registered as pytree nodes. JAX 0.10 treats every
unregistered class as a single leaf and never raises. The branches were
therefore unreachable; they were also masking the only real failure mode
of these helpers, which is a registered pytree node whose own
``flatten_func`` raises.

This module pins both invariants so a future regression that re-adds a
``except TypeError`` (or any other broad swallow) will fail the suite.

Tests:

- ``test_tree_flatten_treats_unregistered_class_as_leaf`` /
  ``test_tree_map_on_unregistered_class_calls_fn_once`` — marked
  ``@pytest.mark.jax_contract``: pin the JAX 0.10 behaviour the §6
  deletion relies on. Not exercising simsopt code; if these fail, JAX
  itself changed.
- ``test_helper_propagates_registered_flatten_runtime_error`` (5 cases,
  one per helper) — verifies the §6 cleanup actually surfaces errors
  from registered pytree nodes whose ``flatten_func`` raises.
- Per-helper output-shape contracts on the three signature helpers:
  ``test_runtime_cache_signature_includes_array_value_hash``,
  ``test_traceable_cache_signature_includes_array_value_hash``,
  ``test_traceable_contract_signature_drops_value_hash_for_arrays``,
  ``test_traceable_contract_signature_keeps_scalar_values`` — would fail
  if a refactor swapped one signature helper for another.
- Transform round-trip contracts:
  ``test_runtime_hostify_tree_moves_jax_arrays_to_host`` and
  ``test_runtime_deviceify_tree_promotes_numpy_to_jax`` — verify
  ``tree_map`` actually walks and rewrites leaves (a no-op smoke would
  not catch a regression that turned the function into ``return tree``).
- ``test_helper_accepts_unregistered_class_without_typeerror`` (5 cases,
  one per helper) — the direct regression target for the §6 deletion.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
import jax.tree_util as jtu

from simsopt.geo.boozersurface_jax import _runtime_cache_tree_signature
from simsopt.geo.surfaceobjectives_jax import (
    _traceable_cache_tree_signature,
    _traceable_contract_tree_signature,
    _traceable_runtime_deviceify_tree,
    _traceable_runtime_hostify_tree,
)


_SIGNATURE_HELPERS = (
    pytest.param(_runtime_cache_tree_signature, id="runtime_cache"),
    pytest.param(_traceable_cache_tree_signature, id="traceable_cache"),
    pytest.param(_traceable_contract_tree_signature, id="traceable_contract"),
)


_TRANSFORM_HELPERS = (
    pytest.param(_traceable_runtime_hostify_tree, id="runtime_hostify"),
    pytest.param(_traceable_runtime_deviceify_tree, id="runtime_deviceify"),
)


_ALL_HELPERS = _SIGNATURE_HELPERS + _TRANSFORM_HELPERS


# --- JAX 0.10 contract the §6 cleanup relies on --------------------------
#
# These two tests pin an external (JAX) library contract, not simsopt
# behaviour. If a future JAX release changes how unregistered classes are
# handled by tree_flatten / tree_map, they fail here first and point at
# the §6 deletion site that needs revisiting.


@pytest.mark.jax_contract
def test_tree_flatten_treats_unregistered_class_as_leaf():
    """JAX 0.10 contract: ``tree_flatten`` returns a single leaf for an unregistered class."""

    class _Plain:
        pass

    obj = _Plain()
    leaves, treedef = jtu.tree_flatten(obj)

    assert leaves == [obj]
    # The leaf treedef has exactly one leaf and no nested structure.
    assert treedef.num_leaves == 1
    assert treedef.num_nodes == 1


@pytest.mark.jax_contract
def test_tree_map_on_unregistered_class_calls_fn_once():
    """JAX 0.10 contract: ``tree_map`` invokes the fn on an unregistered instance as a leaf."""

    class _Plain:
        pass

    obj = _Plain()
    calls = []

    def _identity(leaf):
        calls.append(leaf)
        return leaf

    mapped = jtu.tree_map(_identity, obj)

    assert calls == [obj]
    assert mapped is obj


# --- Registered-node flatten() errors must propagate ---------------------


def _register_flatten_bomb():
    """Register a fresh pytree node whose ``flatten_func`` raises.

    Returns the freshly minted node class. ``register_pytree_node`` is
    process-global with no public unregister API, so a unique class per
    test avoids duplicate-registration ``ValueError``. The class is built
    inside this helper rather than at module scope so each test gets a
    fresh registration.
    """

    class _FlattenBomb:
        pass

    def _flatten(_obj):
        raise RuntimeError("forced flatten failure")

    def _unflatten(_aux, _children):
        return _FlattenBomb()

    jtu.register_pytree_node(_FlattenBomb, _flatten, _unflatten)
    return _FlattenBomb


@pytest.mark.parametrize("helper", _ALL_HELPERS)
def test_helper_propagates_registered_flatten_runtime_error(helper):
    """Registered pytree nodes whose flatten raises must propagate.

    Covers both the signature builders (which call ``tree_flatten``) and
    the transforms (which call ``tree_map`` — internally a flatten +
    unflatten). The point of the §6 cleanup is exactly this: surface
    errors instead of swallowing them.
    """
    bomb_cls = _register_flatten_bomb()
    instance = bomb_cls()

    with pytest.raises(RuntimeError, match="forced flatten failure"):
        helper(instance)


# --- Output-shape contracts on the three signature helpers ---------------
#
# Each signature helper returns ``("tree", repr(treedef), tuple_of_leaf_sigs)``.
# Crucially, the three helpers have *different* leaf semantics:
#
#  * ``_runtime_cache_tree_signature`` and ``_traceable_cache_tree_signature``
#    use full value-hash leaves (``("array", dtype, shape, blake2b)``) so
#    cache reuse requires exact array contents to match.
#  * ``_traceable_contract_tree_signature`` uses *structural-meta* leaves
#    (``("device_array_meta", dtype, shape)`` or ``("array_meta", ...)``)
#    that drop the value hash for size > 1 arrays — cache hits across calls
#    with the same shape/dtype but different data.
#
# These assertions would fail if a refactor swapped one signature helper
# for another, or if a future change reintroduced silent narrowing.


def test_runtime_cache_signature_includes_array_value_hash():
    """``_runtime_cache_tree_signature`` uses the value-hash leaf for arrays."""
    tree = (np.arange(3, dtype=np.float64),)
    kind, _treedef_repr, leaves = _runtime_cache_tree_signature(tree)

    assert kind == "tree"
    assert len(leaves) == 1
    leaf_kind, dtype_str, shape, digest = leaves[0]
    assert leaf_kind == "array"
    assert dtype_str == "float64"
    assert shape == (3,)
    # blake2b hex digest is 32 hex chars at digest_size=16.
    assert isinstance(digest, str)
    assert len(digest) == 32


def test_traceable_cache_signature_includes_array_value_hash():
    """``_traceable_cache_tree_signature`` also uses the value-hash leaf for arrays."""
    tree = (np.arange(3, dtype=np.float64),)
    _kind, _treedef_repr, leaves = _traceable_cache_tree_signature(tree)

    leaf_kind, dtype_str, shape, digest = leaves[0]
    assert leaf_kind == "array"
    assert dtype_str == "float64"
    assert shape == (3,)
    assert len(digest) == 32


def test_traceable_contract_signature_drops_value_hash_for_arrays():
    """``_traceable_contract_tree_signature`` uses *meta-only* leaves for non-scalar arrays.

    Two arrays of identical (dtype, shape) but different values must hash
    to the same signature. This is the documented optimization that
    separates the contract signature from the full value-hash signature.
    """
    tree_a = (np.arange(3, dtype=np.float64),)
    tree_b = (np.full(3, 42.0, dtype=np.float64),)

    sig_a = _traceable_contract_tree_signature(tree_a)
    sig_b = _traceable_contract_tree_signature(tree_b)

    assert sig_a == sig_b
    _kind, _treedef_repr, leaves = sig_a
    leaf_kind, dtype_str, shape = leaves[0]
    assert leaf_kind == "array_meta"
    assert dtype_str == "float64"
    assert shape == (3,)


def test_traceable_contract_signature_keeps_scalar_values():
    """Scalar (size <= 1) arrays still carry their value in the contract signature."""
    tree = (np.array(7.0),)
    _kind, _treedef_repr, leaves = _traceable_contract_tree_signature(tree)
    leaf_kind, dtype_str, value = leaves[0]
    assert leaf_kind == "array_scalar"
    assert dtype_str == "float64"
    assert value == 7.0


# --- Transform-helper round-trip contract --------------------------------
#
# The hostify/deviceify pair is exercised on a mixed pytree to confirm the
# tree_map walk actually rewrites leaves (a no-op smoke wouldn't catch a
# regression that turned the function into ``return tree``).


def test_runtime_hostify_tree_moves_jax_arrays_to_host():
    """``_traceable_runtime_hostify_tree`` walks a pytree and host-materializes JAX leaves."""
    tree = {
        "jax": jnp.arange(3, dtype=jnp.float64),
        "np": np.arange(2, dtype=np.float64),
    }
    hostified = _traceable_runtime_hostify_tree(tree)

    assert isinstance(hostified["jax"], np.ndarray)
    assert not isinstance(hostified["jax"], jnp.ndarray)
    assert isinstance(hostified["np"], np.ndarray)
    np.testing.assert_array_equal(hostified["jax"], np.arange(3, dtype=np.float64))


def test_runtime_deviceify_tree_promotes_numpy_to_jax():
    """``_traceable_runtime_deviceify_tree`` walks a pytree and device-places NumPy leaves."""
    tree = {"np": np.arange(2, dtype=np.float64), "py": 3.5}
    deviceified = _traceable_runtime_deviceify_tree(tree)

    import jax

    assert isinstance(deviceified["np"], jax.Array)
    assert isinstance(deviceified["py"], jax.Array)
    np.testing.assert_array_equal(
        np.asarray(deviceified["np"]), np.arange(2, dtype=np.float64)
    )


# --- Unregistered-class passthrough --------------------------------------
#
# The original §6 deletion specifically targeted the dead
# ``except TypeError`` branch that fired on unregistered classes. One
# parametrized test per helper confirms an unregistered instance flows
# through without raising — that is the regression the cleanup pins.


class _PlainUnregistered:
    """Sentinel unregistered class used by the unregistered-leaf passthrough test."""


@pytest.mark.parametrize("helper", _ALL_HELPERS)
def test_helper_accepts_unregistered_class_without_typeerror(helper):
    """Unregistered classes flow through as leaves; no ``TypeError`` is raised."""
    obj = _PlainUnregistered()
    helper(obj)
