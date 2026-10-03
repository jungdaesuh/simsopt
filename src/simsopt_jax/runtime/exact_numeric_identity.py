"""Dependency-light exact identities for finite numeric pytrees."""

from __future__ import annotations

import hashlib
import json
import math

import jax
import numpy as np

from simsopt_jax.runtime.host_boundary import host_array as _host_array


def exact_numeric_tree_sha256(value: object) -> str:
    """Hash a finite pytree's structure, scalar types, dtypes, shapes, and bytes."""
    leaves, tree_definition = jax.tree_util.tree_flatten(value)
    hasher = hashlib.sha256()
    hasher.update(repr(tree_definition).encode("utf-8"))
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
            array = np.ascontiguousarray(_host_array(leaf))
            if array.dtype.hasobject:
                raise TypeError("Exact identity does not accept object arrays.")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError("Exact identity does not accept non-finite arrays.")
            hasher.update(b"array\0")
            hasher.update(array.dtype.str.encode("utf-8"))
            hasher.update(
                repr(tuple(int(size) for size in array.shape)).encode("utf-8")
            )
            hasher.update(array.tobytes(order="C"))
        elif isinstance(leaf, (str, bool, int, float, type(None))):
            if isinstance(leaf, float) and not math.isfinite(leaf):
                raise ValueError("Exact identity does not accept non-finite scalars.")
            scalar = {"type": type(leaf).__name__, "value": leaf}
            hasher.update(
                json.dumps(
                    scalar,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        else:
            raise TypeError(
                "Exact identity encountered an unsupported leaf: "
                f"{type(leaf).__qualname__}"
            )
    return hasher.hexdigest()
