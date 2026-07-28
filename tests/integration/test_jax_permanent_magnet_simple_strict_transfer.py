"""Strict-transfer coverage for the exact permanent-magnet workflow."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    _build_cpu_grid,
    _scale_configuration,
)
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_permanent_magnet_example_has_one_batched_numerical_host_boundary() -> None:
    source = (
        REPOSITORY_ROOT / "examples/jax/1_Simple/permanent_magnet_simple.py"
    ).read_text()

    assert source.count("jax.device_get(") == 1


def test_permanent_magnet_gpmo_keeps_numerical_workflow_on_device() -> None:
    cpu_grid = _build_cpu_grid(_scale_configuration("bounded"))
    grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)

    with jax.transfer_guard("disallow"):
        result = GPMO_baseline_jax(grid, K=40)

    moments, residual, selected, target = jax.device_get(
        (result.m, result.residual, result.selected_dipoles, grid.b_obj)
    )
    initial_residual = np.asarray(target, dtype=np.float64)
    assert selected.shape == (40,)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == 40
    assert float(np.vdot(residual, residual)) < float(
        np.vdot(initial_residual, initial_residual)
    )
