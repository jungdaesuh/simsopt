"""Memory-bounded result coverage for the baseline JAX GPMO solver."""

from __future__ import annotations

import jax
import numpy as np
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    _build_cpu_grid,
    _scale_configuration,
)
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax


def test_permanent_magnet_example_can_omit_iteration_state_history() -> None:
    cpu_grid = _build_cpu_grid(_scale_configuration("bounded"))
    grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)

    result = GPMO_baseline_jax(grid, K=40, retain_history=False)
    moments, selected = jax.device_get((result.m, result.selected_dipoles))

    assert result.x_history.shape == (0, grid.ndipoles, 3)
    assert result.m_history.shape == (0, grid.ndipoles, 3)
    assert result.residual_history.shape == (0,)
    assert selected.shape == (40,)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == 40
