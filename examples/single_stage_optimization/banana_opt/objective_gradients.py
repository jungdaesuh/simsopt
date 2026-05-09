from __future__ import annotations

import numpy as np


def objective_gradient(objective, objective_optimizable=None):
    if objective_optimizable is None:
        return np.asarray(objective.dJ(), dtype=float)
    try:
        partial_gradient = objective.dJ(partials=True)
    except TypeError:
        return np.asarray(objective.dJ(), dtype=float)
    if callable(partial_gradient):
        return np.asarray(partial_gradient(objective_optimizable), dtype=float)
    return np.asarray(partial_gradient, dtype=float)
