"""Shared host-array contract checks."""

from __future__ import annotations

import numpy as np

_INT32_MAX = int(np.iinfo(np.int32).max)


def require_nonnegative_int32_indices(name: str, values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must contain integer indices; got {array.dtype}.")
    if array.size == 0:
        return array
    min_value = int(np.min(array))
    max_value = int(np.max(array))
    if min_value < 0 or max_value > _INT32_MAX:
        raise ValueError(
            f"{name} indices must be in [0, {_INT32_MAX}] before int32 staging; "
            f"got range [{min_value}, {max_value}]."
        )
    return array
