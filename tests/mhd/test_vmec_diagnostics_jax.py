import numpy as np
import jax

from simsopt._core.optimizable import Optimizable
from simsopt.jax_core.mhd_reductions import (
    iota_target_metric_j,
    iota_weighted_j,
    well_weighted_j,
)
from simsopt.mhd.vmec_diagnostics import IotaTargetMetric, IotaWeighted, WellWeighted


class _FrozenVmec(Optimizable):
    def __init__(self, s_half_grid, iotas_half_grid, vp_half_grid, ds):
        self.s_half_grid = np.asarray(s_half_grid, dtype=np.float64)
        self.ds = float(ds)
        self.wout = type(
            "FrozenWout",
            (),
            {
                "iotas": np.concatenate(
                    (np.array([np.nan], dtype=np.float64), iotas_half_grid)
                ),
                "vp": np.concatenate(
                    (np.array([np.nan], dtype=np.float64), vp_half_grid)
                ),
            },
        )()
        self.boundary = Optimizable()
        super().__init__()

    def run(self):
        pass


def _target_function(s):
    return np.cos(s)


def _edge_weight(s):
    return np.exp(-((s - 0.5) ** 2) / 0.2**2)


def _axis_weight(s):
    return np.exp(-(s**2) / 0.4**2)


def _boundary_weight(s):
    return np.exp(-((1.0 - s) ** 2) / 0.4**2)


def test_iota_target_metric_reducer_matches_cpu_object_oracle():
    s_half_grid = np.linspace(0.1, 0.9, 5)
    iotas = 0.4 + 0.2 * s_half_grid
    ds = 0.125
    vmec = _FrozenVmec(s_half_grid, iotas, np.ones_like(iotas), ds)

    actual = iota_target_metric_j(iotas, _target_function(s_half_grid), ds)
    expected = IotaTargetMetric(vmec, _target_function).J()

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_iota_weighted_reducer_matches_cpu_object_oracle():
    s_half_grid = np.linspace(0.1, 0.9, 5)
    iotas = 0.6 + 0.1 * s_half_grid
    weights = _edge_weight(s_half_grid)
    vmec = _FrozenVmec(s_half_grid, iotas, np.ones_like(iotas), ds=0.125)

    actual = iota_weighted_j(iotas, weights)
    expected = IotaWeighted(vmec, _edge_weight).J()

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_well_weighted_reducer_matches_cpu_object_oracle():
    s_half_grid = np.linspace(0.1, 0.9, 5)
    vp = 1.0 + 0.3 * s_half_grid + 0.1 * s_half_grid**2
    weights1 = _axis_weight(s_half_grid)
    weights2 = _boundary_weight(s_half_grid)
    vmec = _FrozenVmec(s_half_grid, np.ones_like(vp), vp, ds=0.125)

    actual = well_weighted_j(vp, weights1, weights2)
    expected = WellWeighted(vmec, _axis_weight, _boundary_weight).J()

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_vmec_scalar_reducers_trace_under_jit():
    s_half_grid = np.linspace(0.1, 0.9, 5)
    iotas = 0.6 + 0.1 * s_half_grid
    target = _target_function(s_half_grid)
    weights1 = _axis_weight(s_half_grid)
    weights2 = _boundary_weight(s_half_grid)
    vp = 1.0 + 0.3 * s_half_grid + 0.1 * s_half_grid**2
    vmec = _FrozenVmec(s_half_grid, iotas, vp, ds=0.125)

    metric = jax.jit(iota_target_metric_j)(iotas, target, 0.125)
    weighted = jax.jit(iota_weighted_j)(iotas, weights1)
    well = jax.jit(well_weighted_j)(vp, weights1, weights2)

    np.testing.assert_allclose(
        np.asarray(metric), IotaTargetMetric(vmec, _target_function).J()
    )
    np.testing.assert_allclose(
        np.asarray(weighted), IotaWeighted(vmec, _axis_weight).J()
    )
    np.testing.assert_allclose(
        np.asarray(well),
        WellWeighted(vmec, _axis_weight, _boundary_weight).J(),
    )
