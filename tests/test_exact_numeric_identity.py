"""Tests for exact finite-numeric pytree identities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256


def test_exact_numeric_identity_matches_equal_numeric_trees() -> None:
    left = {
        "state": jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        "factors": (
            np.asarray([[3.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
        ),
    }
    right = {
        "state": jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        "factors": (
            np.asarray([[3.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
        ),
    }

    assert exact_numeric_tree_sha256(left) == exact_numeric_tree_sha256(right)


@pytest.mark.parametrize(
    "changed",
    (
        {"state": np.asarray([1.0, -2.5], dtype=np.float64)},
        {"state": np.asarray([1.0, -2.0], dtype=np.float32)},
        {"state": np.asarray([[1.0, -2.0]], dtype=np.float64)},
        {"state": (np.asarray([1.0, -2.0], dtype=np.float64),)},
    ),
)
def test_exact_numeric_identity_binds_bytes_dtype_shape_and_structure(
    changed: object,
) -> None:
    baseline = {"state": np.asarray([1.0, -2.0], dtype=np.float64)}

    assert exact_numeric_tree_sha256(baseline) != exact_numeric_tree_sha256(changed)


def test_exact_numeric_identity_binds_scalar_type() -> None:
    assert exact_numeric_tree_sha256(1) != exact_numeric_tree_sha256(1.0)
    assert exact_numeric_tree_sha256(False) != exact_numeric_tree_sha256(0)


@pytest.mark.parametrize(
    "unsafe",
    (
        np.asarray([object()], dtype=object),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([np.inf], dtype=np.float64),
        float("nan"),
        float("inf"),
    ),
)
def test_exact_numeric_identity_rejects_unsafe_leaves(unsafe: object) -> None:
    error_type = (
        TypeError
        if isinstance(unsafe, np.ndarray) and unsafe.dtype.hasobject
        else ValueError
    )

    with pytest.raises(error_type):
        exact_numeric_tree_sha256({"unsafe": unsafe})


def test_exact_numeric_identity_rejects_unsupported_leaf() -> None:
    with pytest.raises(TypeError, match="unsupported leaf"):
        exact_numeric_tree_sha256({"unsafe": object()})
