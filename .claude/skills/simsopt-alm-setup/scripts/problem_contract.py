"""The row checks of the problem-module contract (``references/api.md``) that
``gradient_check.py`` and ``sign_check.py`` share: they key each row by its
name, so a name that repeats, or a physics whose row count differs from the
names, would let a check read one row and report it as another.

``checked_constraint_names(names)`` returns ``names`` (the problem's
``constraint_names``) as a tuple after checking that every name is a non-empty
string and no name repeats; ``checked_constraint_values(physics, names)``
checks the names the same way and returns the physics' constraint values
after checking that it has one value and one gradient per name. Both raise
``ValueError`` naming the breach, before any row is read by name.
"""

from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np


def checked_constraint_names(names) -> Tuple[str, ...]:
    names = tuple(names)
    invalid = [name for name in names if not isinstance(name, str) or not name]
    if invalid:
        raise ValueError(f"problem.constraint_names must be non-empty strings, got {invalid!r}")
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"problem.constraint_names must be unique (one name per row); "
                         f"repeated: {repeated!r}")
    return names


def checked_constraint_values(physics, names) -> np.ndarray:
    names = checked_constraint_names(names)
    values = np.asarray(physics.constraint_values, dtype=float)
    if values.ndim != 1 or len(values) != len(names) or len(physics.constraint_grads) != len(names):
        raise ValueError(f"problem.physics(x) returned {values.size} constraint values and "
                         f"{len(physics.constraint_grads)} constraint gradients for {len(names)} "
                         f"constraint_names; each row needs one name, one value and one gradient")
    return values
