"""The GPMO carries hold no host buffer.

Coordinator flag from unit ``rcls-gsco``: ``runtime_init_array`` builds a HOST
``np.full(...)`` and places it with ``jax.device_put``; inside a traced loop
that becomes a staged host-to-device copy, which the parity intent's
``transfer_guard_host_to_device("disallow")`` (``arbiter.py:148``) rejects --
the GPU-only "Disallowed host-to-device transfer" that killed the GSCO
multistep lane.

The GPMO lanes do not share the pattern. The shipped kernels
(``simsopt_jax.solve.permanent_magnet`` -> ``simsopt_jax/core/pm_optimization.py``)
build every carry buffer with ``jnp.zeros``, which is device-native, and the
one module that does use the helper, ``simsopt_jax/core/pm_workflow.py``, is
imported by ``tests/solve/test_pm_workflow_jax.py`` alone and exposes its
``pm_gpmo_*_initial_state`` builders as eager entry points (an explicit
transfer, which the guard allows). This test holds that negative: it fails on
CPU if any GPMO carry gains a host buffer.
"""

from __future__ import annotations

import jax
import numpy as np
from examples.jax.parity.cases.native_permanent_magnet_muse import (
    _build_cpu_grid,
    _scale_configuration,
)
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import (
    GPMO_ArbVec_backtracking_jax,
    GPMO_baseline_jax,
)

BASELINE_STEPS = 10


def _device_grid() -> PermanentMagnetGridJAX:
    """The bounded MUSE grid, placed before the guard is armed."""
    return jax.device_put(
        PermanentMagnetGridJAX.from_cpu(
            _build_cpu_grid(_scale_configuration("bounded"))
        )
    )


def test_arbvec_backtracking_runs_without_a_host_to_device_transfer() -> None:
    configuration = _scale_configuration("bounded")
    grid = _device_grid()

    with jax.transfer_guard_host_to_device("disallow"):
        result = GPMO_ArbVec_backtracking_jax(
            grid,
            K=configuration["iterations"],
            Nadjacent=configuration["adjacent_count"],
            backtracking=configuration["backtracking"],
            thresh_angle=configuration["threshold_angle"],
            max_nMagnets=configuration["max_magnets"],
            record_every=int(configuration["iterations"])
            // int(configuration["history_count"]),
        )
        jax.block_until_ready(result.m)

    moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    assert (
        int(np.count_nonzero(np.linalg.norm(moments, axis=1)))
        == (configuration["max_magnets"])
    )


def test_baseline_gpmo_runs_without_a_host_to_device_transfer() -> None:
    grid = _device_grid()

    with jax.transfer_guard_host_to_device("disallow"):
        result = GPMO_baseline_jax(grid, K=BASELINE_STEPS)
        jax.block_until_ready(result.m)

    moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    assert int(np.count_nonzero(np.linalg.norm(moments, axis=1))) == BASELINE_STEPS
