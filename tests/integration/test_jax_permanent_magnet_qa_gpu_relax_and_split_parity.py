"""GPU nphi=64 two-outer QA relax-and-split moment parity vs native."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import jax
import numpy as np
import pytest

from benchmarks.pm_gpmo_probes import GridSpec, _build_qa_grid
from conftest import enable_strict_parity_backend, host_array, parity_default_device
from simsopt.solve.permanent_magnet_optimization import relax_and_split
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import relax_and_split_jax


def test_qa_nphi64_two_outer_gpu_moments_match_native(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    spec = GridSpec(
        nphi=64,
        ntheta=64,
        downsample=None,
        coordinate_flag="cylindrical",
        dr=0.02,
        inner_offset=0.05,
        outer_offset=0.15,
        source="qa gpu two-outer parity nphi=64",
    )
    with redirect_stdout(io.StringIO()):
        grid = _build_qa_grid(spec).grid
    staged = PermanentMagnetGridJAX.from_cpu(grid)
    jax.block_until_ready((staged.A_obj, staged.b_obj, staged.ATb))
    with redirect_stdout(io.StringIO()):
        reg_l0, _, _, nu = grid.rescale_for_opt(0.05, 0.0, 0.0, 1.0e10)
    alpha = 2.0 * (1.0 - 1.0e-5) / float(grid.ATA_scale)
    ndipoles = int(grid.ndipoles)
    initial = np.zeros(ndipoles * 3, dtype=np.float64)
    kwargs = {
        "nu": nu,
        "max_iter": 10,
        "max_iter_RS": 2,
        "reg_l0": reg_l0,
        "epsilon": 0.0,
        "epsilon_RS": 0.0,
        "min_fb": 0.0,
        "verbose": True,
    }
    with redirect_stdout(io.StringIO()):
        relax_and_split(grid, m0=initial, **kwargs)
    native_m = np.asarray(grid.m, dtype=np.float64).reshape((ndipoles, 3))
    with parity_default_device("gpu"):
        result = relax_and_split_jax(
            staged,
            np.zeros((ndipoles, 3), dtype=np.float64),
            alpha=alpha,
            max_iter=10,
            max_iter_RS=2,
            nu=nu,
            reg_l0=reg_l0,
            epsilon=0.0,
            epsilon_RS=0.0,
        )
        jax.block_until_ready(result.m)
        jax_m = host_array(result.m, dtype=np.float64)
    scale = max(float(np.max(np.abs(native_m))), 1.0e-30)
    rel = float(np.max(np.abs(jax_m - native_m)) / scale)
    assert rel <= 1.0e-9, rel
